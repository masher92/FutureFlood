#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # Registers the .rio accessor used by xarray.
import xarray as xr
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

# --------------------------
# Configuration
# --------------------------
CRS_TARGET = "EPSG:27700"

DEM_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/Model_builds/Pluvial/v4/dem/"
CSV_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall_events/fixed_threshold_30mm/"
NETCDF_DIR = "/scratch/hydro4/shared_data/climate_projections/UKCP18/UKCP_local/Soil_moisture/5km_regridded/"
SHAPEFILE_PATH = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/CEH_catchments/CEH_IHU_with_coastline/hyd_areas_GB_with_subcatchments.shp"
SOIL_TEXTURE_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/Model_builds/Pluvial/v4/soil_texture/"
POROSITY_CSV = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/SoilTexture/USDA_soil_texture_effective_porosity.csv"

# Both output types are written below this directory:
#   <OUTPUT_VOL_DIR>/<ensemble>/tif/<HA_NUM>/*.tif
#   <OUTPUT_VOL_DIR>/<ensemble>/asc/<HA_NUM>/*.asc
OUTPUT_VOL_DIR = (
    "/scratch/hydro5/users/la17355/FUTURE-FLOOD/UKCP_soil_moisture/"
    "volumetric_30m/soil_depth_missing_tests/"
)

LOG_DIR = "/scratch/hydro4/users/kv25483/FutureFlood/UKCP_soil_moisture/"

DRY_RUN = False
MAX_WORKERS = min(8, os.cpu_count() or 1)
print("Hey")

# --------------------------
# Helper functions
# --------------------------
def extract_ha_num(path_or_name):
    """Extract an HA identifier such as 27, 27_a, or 54_d."""
    base = os.path.splitext(os.path.basename(path_or_name))[0]
    match = re.search(r"(\d+(?:_[A-Za-z0-9]+)?)$", base)
    return match.group(1) if match else None


def load_catchments(shapefile_path):
    return gpd.read_file(shapefile_path)


def find_matching_netcdf(netcdf_dir, year, month, day):
    """Return the NetCDF whose date range contains the event date."""
    target_date_int = int(f"{year:04d}{month:02d}{day:02d}")

    for nc_file in glob.glob(os.path.join(netcdf_dir, "*.nc")):
        match = re.search(r"_(\d{8})-(\d{8})_", os.path.basename(nc_file))
        if match and int(match.group(1)) <= target_date_int <= int(match.group(2)):
            return nc_file

    logging.warning(
        "[NetCDF Search Fail] No match for date %s in %s",
        target_date_int,
        netcdf_dir,
    )
    return None


def clip_gravitational(ds, var, bounds):
    """Clip gravitational soil-moisture data in memory."""
    if not ds.rio.crs:
        ds = ds.rio.write_crs(CRS_TARGET)

    return ds[var].rio.clip_box(
        minx=bounds.bounds[0],
        miny=bounds.bounds[1],
        maxx=bounds.bounds[2],
        maxy=bounds.bounds[3],
    )


def resample_to_30m_array(source_data, dem):
    """Resample clipped gravitational data to the corresponding DEM grid."""
    source_array = np.squeeze(source_data.values)
    if source_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D gravitational-data array, got {source_array.shape}"
        )

    resampled = np.empty((dem["height"], dem["width"]), dtype=source_array.dtype)

    reproject(
        source=source_array,
        destination=resampled,
        src_transform=source_data.rio.transform(),
        src_crs=source_data.rio.crs,
        dst_transform=dem["transform"],
        dst_crs=dem["crs"],
        resampling=Resampling.nearest,
    )

    out_meta = {
        "driver": "GTiff",
        "dtype": source_array.dtype,
        "nodata": -9999,
        "width": dem["width"],
        "height": dem["height"],
        "count": 1,
        "crs": dem["crs"],
        "transform": dem["transform"],
        "compress": "LZW",
        "tiled": True,
    }
    return resampled, out_meta


def convert_to_volumetric_array(
    grav_data,
    grav_meta,
    vol_tif_out,
    vol_asc_out,
    porosity_data,
    dem_mask,
):
    """Convert soil moisture and write matching GeoTIFF and ASCII Grid files."""
    try:
        dem_mask_bool = dem_mask == -9999

        # Updated gravitational-to-volumetric equation.
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_sm = ((grav_data / 1000.0) / 0.225) / porosity_data
            vol_sm = np.clip(vol_sm, 0, 1)
            vol_sm = np.round(vol_sm, 2)

        vol_sm[dem_mask_bool] = -9999
        output_array = vol_sm.astype(np.float32)

        tif_meta = grav_meta.copy()
        tif_meta.update(
            dtype=rasterio.float32,
            nodata=-9999,
            compress="LZW",
            driver="GTiff",
        )
        os.makedirs(os.path.dirname(vol_tif_out), exist_ok=True)
        with rasterio.open(vol_tif_out, "w", **tif_meta) as dst:
            dst.write(output_array, 1)

        # AAIGrid does not support GeoTIFF compression or tiling options.
        asc_meta = tif_meta.copy()
        asc_meta.pop("compress", None)
        asc_meta.pop("tiled", None)
        asc_meta.update(driver="AAIGrid")
        os.makedirs(os.path.dirname(vol_asc_out), exist_ok=True)
        with rasterio.open(vol_asc_out, "w", **asc_meta) as dst:
            dst.write(output_array, 1)

    except Exception as exc:
        logging.error("[Volumetric Conversion Error] %s: %s", vol_tif_out, exc)


def load_porosity_grids(porosity_csv, soil_texture_dir, target_has):
    """Load porosity values mapped from each target soil-texture raster."""
    df = pd.read_csv(porosity_csv)
    porosity_map = df.set_index("USDA_soil")["Effective_porosity"].to_dict()
    soil_grids = {}

    for tif in glob.glob(os.path.join(soil_texture_dir, "*.tif")):
        ha_num = extract_ha_num(tif)
        if ha_num is None or ha_num not in target_has:
            continue

        with rasterio.open(tif) as src:
            data = src.read(1)
            porosity = np.vectorize(lambda value: porosity_map.get(value, np.nan))(data)
            porosity[data == src.nodata] = np.nan
            soil_grids[ha_num] = {"grid": porosity}

    return soil_grids


def process_event(row, dem_path, ensemble, porosity_grids, dem_cached, target_has):
    """Process one rainfall event for one catchment."""
    try:
        year = int(row["start_year"])
        month = int(row["start_month"])
        day = int(row["start_day"])
        event_num = int(row["event_num"])
        start_index = int(row["start_indices"])
        ha_num = extract_ha_num(dem_path)

        if ha_num is None:
            logging.warning("[DEM Parse Fail] Could not extract HA from %s", dem_path)
            return
        if ha_num not in target_has:
            return

        nc_path = find_matching_netcdf(
            os.path.join(NETCDF_DIR, ensemble), year, month, day
        )
        if not nc_path:
            logging.error(
                "[NetCDF Missing] Event %s %s-%s-%s in %s",
                event_num,
                year,
                month,
                day,
                ensemble,
            )
            return
        if ha_num not in dem_cached:
            logging.error("[DEM Missing] %s not found in dem_cached", ha_num)
            return
        if ha_num not in porosity_grids:
            logging.error("[Porosity Missing] %s for event %s", ha_num, event_num)
            return

        dem_info = dem_cached[ha_num]
        event_base = f"soil_{ha_num}_{event_num}_{year}_{month}_{day}_{ensemble}"
        vol_tif = os.path.join(
            OUTPUT_VOL_DIR, ensemble, "tif", ha_num, f"{event_base}_vol_30m.tif"
        )
        vol_asc = os.path.join(
            OUTPUT_VOL_DIR, ensemble, "asc", ha_num, f"{event_base}_vol_30m.asc"
        )

        with xr.open_dataset(nc_path) as ds:
            index = (start_index - 1) // 24
            grav_clipped = clip_gravitational(
                ds.isel(time=index),
                "moisture_content_of_soil_layer",
                dem_info["bounds"],
            )
            grav_30m_array, grav_30m_meta = resample_to_30m_array(
                grav_clipped, dem_info
            )

        convert_to_volumetric_array(
            grav_30m_array,
            grav_30m_meta,
            vol_tif,
            vol_asc,
            porosity_grids[ha_num]["grid"],
            dem_info["mask"],
        )

    except Exception as exc:
        logging.error(
            "[Processing Error] Event %s: %s", row.get("event_num", "unknown"), exc
        )


def main():
    print("Insidemain")
    parser = argparse.ArgumentParser(
        description="Run soil-moisture pipeline for specific HA_NUM(s)."
    )
    parser.add_argument(
        "target_has",
        nargs="+",
        help="HA_NUM values to process, e.g. 27 27_a 54_b 39_d",
    )
    args = parser.parse_args()
    target_has = set(args.target_has)
    total_start = time.time()

    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(
            LOG_DIR, f"soil_pipeline_{'_'.join(sorted(target_has))}_{log_time}.txt"
        ),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    porosity_grids = load_porosity_grids(POROSITY_CSV, SOIL_TEXTURE_DIR, target_has)
    gdf = load_catchments(SHAPEFILE_PATH)
    gdf["HA_NAME_NORM"] = gdf["HA_NAME"].str.replace(" ", "", regex=False)
    target_catchments = gdf[gdf["HA_NUM"].astype(str).isin(target_has)]
    if target_catchments.empty:
        logging.error("No catchments found for HA_NUM(s): %s", ", ".join(target_has))
        sys.exit(1)

    target_names_norm = target_catchments["HA_NAME_NORM"].unique().tolist()
    csv_files = []
    print(target_names_norm)
    
    for csv_file in glob.glob(os.path.join(CSV_DIR, "*.csv")):
        basename = os.path.basename(csv_file)
        if any(
            re.search(r"(^|_)" + re.escape(name) + r"(_|$)", basename)
            for name in target_names_norm
        ):
            csv_files.append(csv_file)
    if not csv_files:
        logging.warning("No matching CSV files found.")
        sys.exit(0)
    print(csv_files)

    ensembles = {
        name
        for name in os.listdir(NETCDF_DIR)
        if os.path.isdir(os.path.join(NETCDF_DIR, name))
    }
    dem_files = [
        path
        for path in glob.glob(os.path.join(DEM_DIR, "*.tif"))
        if extract_ha_num(path) in target_has
    ]
    print(dem_files)
    dem_cached = {}
    for dem_path in dem_files:
        ha_num = extract_ha_num(dem_path)
        print(ha_num)
        with rasterio.open(dem_path) as dem:
            dem_cached[ha_num] = {
                "bounds": box(*dem.bounds),
                "transform": dem.transform,
                "crs": dem.crs,
                "width": dem.width,
                "height": dem.height,
                "mask": dem.read(1),
            }

    for csv_path in csv_files:
        csv_name = os.path.basename(csv_path)
        print(csv_name)
        match = re.search(r"_(\d{2})_", csv_name)
        if not match:
            logging.warning("[CSV Skipped] Could not identify ensemble for %s", csv_name)
            continue

        ensemble = f"Ens_{match.group(1)}"
        if ensemble not in ensembles:
            logging.warning("[CSV Skipped] Ensemble %s is unavailable.", ensemble)
            continue

        events_df = pd.read_csv(csv_path)
        logging.info("[CSV Matched] %s | Ensemble: %s", csv_name, ensemble)

        for dem_path in dem_files:
            ha_num = extract_ha_num(dem_path)
            os.makedirs(os.path.join(OUTPUT_VOL_DIR, ensemble, "tif", ha_num), exist_ok=True)
            os.makedirs(os.path.join(OUTPUT_VOL_DIR, ensemble, "asc", ha_num), exist_ok=True)

        if DRY_RUN:
            for dem_path in dem_files:
                ha_num = extract_ha_num(dem_path)
                for _, row in events_df.iterrows():
                    event_base = (
                        f"soil_{ha_num}_{int(row['event_num'])}_{int(row['start_year'])}_"
                        f"{int(row['start_month'])}_{int(row['start_day'])}_{ensemble}"
                    )
                    print(f"INFO {csv_name} | DEM {ha_num}")
                    print(os.path.join(OUTPUT_VOL_DIR, ensemble, "tif", ha_num, f"{event_base}_vol_30m.tif"))
                    print(os.path.join(OUTPUT_VOL_DIR, ensemble, "asc", ha_num, f"{event_base}_vol_30m.asc"))
            continue

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_event,
                    row,
                    dem_path,
                    ensemble,
                    porosity_grids,
                    dem_cached,
                    target_has,
                )
                for dem_path in dem_files
                for _, row in events_df.iterrows()
            ]
            for future in as_completed(futures):
                future.result()

    logging.info("[Pipeline Finished] Total time: %.2fs", time.time() - total_start)


if __name__ == "__main__":
    main()
