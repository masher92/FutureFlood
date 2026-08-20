"""
FUTURE-FLOOD soil moisture pipeline.

For each rainfall event (one row per catchment/ensemble CSV), this script:
  1. Reads UKCP18 gravitational soil moisture (moisture_content_of_soil_layer)
     for the matching date, clips it to the catchment DEM extent, and
     resamples it to the 30 m DEM grid.
  2. Converts that to relative saturation (0-1) using the 0.225 m soil
     layer depth and each 30 m pixel's own effective porosity.
  3. Derives IMD (initial moisture deficit) and remaining upper-zone
     capacity at event outset (IMD * Lu), using a precomputed, static 30 m
     Lu (effective upper-zone depth) raster.
  4. Writes vol_sm, imd, and remaining_capacity as three variables in a
     single combined netCDF file per event.

Lu and porosity are static soil properties -- they don't vary by event or
ensemble, so both are loaded and clipped to each catchment's grid ONCE,
outside the per-event loop, and reused across every event/ensemble that
touches that catchment.
"""
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

# Static 30 m soil rasters (texture-only, don't vary by event).
STATIC_LU_PATH = "../../Data/SoilStaticVariables/Mean_EffectiveDepth_GB_30m.nc"
# Not used by the production pipeline below -- remaining_capacity is
# computed locally from porosity_data * Lu. Kept here only for reference /
# use by a separate QA check script.
STATIC_FUMAX_PATH = "../../Data/SoilStaticVariables/Mean_FUmax_GB_30m.nc"

# Single combined-netCDF output root:
#   <OUTPUT_VOL_DIR>/<ensemble>/nc/<HA_NUM>/*.nc
# NOTE: this currently points at kv25483's scratch space, not la17355's --
# worth double-checking this is deliberate before running at scale.
OUTPUT_VOL_DIR = "/scratch/hydro4/users/kv25483/FutureFlood/Data/SoilDynamicVariables/volume/"

LOG_DIR = "/scratch/hydro4/users/kv25483/FutureFlood"

DRY_RUN = False
MAX_WORKERS = min(8, os.cpu_count() or 1)


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
    """Resample a clipped xarray DataArray to the corresponding DEM grid.

    Used both for the daily gravitational moisture data (per event) and
    for the static Lu raster (once per catchment).
    """
    source_array = np.squeeze(source_data.values)
    if source_array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {source_array.shape}")

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


def load_porosity_grids(porosity_csv, soil_texture_dir, target_has):
    """Load effective porosity, mapped from each target 30 m soil-texture raster."""
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
            soil_grids[ha_num] = {
                "grid": porosity,
                "transform": src.transform,
                "crs": src.crs,
            }
    return soil_grids


def load_static_lu_grids(lu_path, dem_cached, target_has):
    """Clip the national static Lu (effective upper-zone depth, metres)
    raster onto each catchment's own DEM grid, once, for reuse across all
    events/ensembles for that catchment.
    """
    lu_national = rioxarray.open_rasterio(lu_path, masked=True).squeeze()
    if not lu_national.rio.crs:
        lu_national = lu_national.rio.write_crs(CRS_TARGET)

    lu_grids = {}
    for ha_num, dem_info in dem_cached.items():
        if ha_num not in target_has:
            continue
        clipped = lu_national.rio.clip_box(*dem_info["bounds"].bounds)
        lu_30m, _ = resample_to_30m_array(clipped, dem_info)
        lu_grids[ha_num] = lu_30m

    return lu_grids


def array_to_dataarray(array, transform, crs, name, nodata=-9999):
    """Wrap a plain numpy array into a georeferenced xr.DataArray, computing
    pixel-center x/y coordinates from the affine transform."""
    height, width = array.shape
    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    da = xr.DataArray(array, dims=("y", "x"), coords={"y": ys, "x": xs}, name=name)
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(nodata, inplace=True)
    return da


def convert_to_volumetric_array(grav_data, grav_meta, vol_nc_out, porosity_data, lu_data, dem_mask):
    """Convert gravitational soil moisture to relative saturation, IMD, and
    remaining upper-zone capacity, writing all three as variables in a
    single combined netCDF file.
    """
    try:
        dem_mask_bool = dem_mask == -9999
        nodata = -9999

        with np.errstate(divide="ignore", invalid="ignore"):
            vol_sm = ((grav_data / 1000.0) / 0.225) / porosity_data
            vol_sm = np.clip(vol_sm, 0, 1)

            imd = porosity_data * (1.0 - vol_sm)
            remaining_capacity = imd * lu_data

            vol_sm = np.round(vol_sm, 2)
            imd = np.round(imd, 4)
            remaining_capacity = np.round(remaining_capacity, 4)

        vol_sm[dem_mask_bool] = nodata
        imd[dem_mask_bool] = nodata
        remaining_capacity[dem_mask_bool] = nodata

        transform = grav_meta["transform"]
        crs = grav_meta["crs"]

        ds = xr.Dataset(
            {
                "vol_sm": array_to_dataarray(vol_sm.astype(np.float32), transform, crs, "vol_sm", nodata),
                "imd": array_to_dataarray(imd.astype(np.float32), transform, crs, "imd", nodata),
                "remaining_capacity": array_to_dataarray(
                    remaining_capacity.astype(np.float32), transform, crs, "remaining_capacity", nodata
                ),
            }
        )
        ds["vol_sm"].attrs["units"] = "1"  # dimensionless, 0-1
        ds["imd"].attrs["units"] = "1"
        ds["remaining_capacity"].attrs["units"] = "m"

        os.makedirs(os.path.dirname(vol_nc_out), exist_ok=True)
        ds.to_netcdf(vol_nc_out)

    except Exception as exc:
        logging.error("[Volumetric Conversion Error] %s: %s", vol_nc_out, exc)


def process_event(row, dem_path, ensemble, porosity_grids, lu_grids, dem_cached, target_has):
    """Process one rainfall event for one catchment/ensemble."""
    try:
        year = int(row["start_year"])
        month = int(row["start_month"])
        day = int(row["start_day"])
        event_num = int(row["event_num"])
        start_index = int(row["start_indices"])
        ha_num = extract_ha_num(dem_path)

        if ha_num is None or ha_num not in target_has:
            return

        nc_path = find_matching_netcdf(os.path.join(NETCDF_DIR, ensemble), year, month, day)
        if not nc_path:
            logging.error(
                "[NetCDF Missing] Event %s %s-%s-%s in %s", event_num, year, month, day, ensemble
            )
            return
        if ha_num not in dem_cached or ha_num not in porosity_grids or ha_num not in lu_grids:
            logging.error("[Static Grid Missing] %s for event %s", ha_num, event_num)
            return

        dem_info = dem_cached[ha_num]
        event_base = f"soil_{ha_num}_{event_num}_{year}_{month}_{day}_{ensemble}"
        vol_nc = os.path.join(OUTPUT_VOL_DIR, ensemble, "nc", ha_num, f"{event_base}_vol_30m.nc")

        with xr.open_dataset(nc_path) as ds:
            index = (start_index - 1) // 24
            grav_clipped = clip_gravitational(
                ds.isel(time=index), "moisture_content_of_soil_layer", dem_info["bounds"]
            )
            grav_30m_array, grav_30m_meta = resample_to_30m_array(grav_clipped, dem_info)

        convert_to_volumetric_array(
            grav_30m_array,
            grav_30m_meta,
            vol_nc,
            porosity_grids[ha_num]["grid"],
            lu_grids[ha_num],
            dem_info["mask"],
        )

    except Exception as exc:
        logging.error("[Processing Error] Event %s: %s", row.get("event_num", "unknown"), exc)


def main():
    parser = argparse.ArgumentParser(
        description="Run soil-moisture pipeline (combined vol_sm/IMD/capacity netCDF) for specific HA_NUM(s)."
    )
    parser.add_argument("target_has", nargs="+", help="HA_NUM values to process, e.g. 27 27_a 54_b 39_d")
    args = parser.parse_args()
    run_pipeline(set(args.target_has))


def run_pipeline(target_has):
    """Core pipeline logic, separated from argparse so it can be called
    directly from a notebook: run_pipeline({"27", "54_b"})
    """
    total_start = time.time()

    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, f"soil_pipeline_{'_'.join(sorted(target_has))}_{log_time}.txt"),
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
    for csv_file in glob.glob(os.path.join(CSV_DIR, "*.csv")):
        basename = os.path.basename(csv_file)
        if any(
            re.search(r"(^|_)" + re.escape(name) + r"(_|$)", basename)
            for name in target_names_norm
        ):
            csv_files.append(csv_file)
    if not csv_files:
        logging.warning("No matching CSV files found.")
        return

    ensembles = {
        name for name in os.listdir(NETCDF_DIR) if os.path.isdir(os.path.join(NETCDF_DIR, name))
    }
    dem_files = [
        path for path in glob.glob(os.path.join(DEM_DIR, "*.tif")) if extract_ha_num(path) in target_has
    ]

    dem_cached = {}
    for dem_path in dem_files:
        ha_num = extract_ha_num(dem_path)
        with rasterio.open(dem_path) as dem:
            dem_cached[ha_num] = {
                "bounds": box(*dem.bounds),
                "transform": dem.transform,
                "crs": dem.crs,
                "width": dem.width,
                "height": dem.height,
                "mask": dem.read(1),
            }

    # Static soil rasters -- loaded/clipped ONCE per catchment, reused for
    # every event and every ensemble below.
    lu_grids = load_static_lu_grids(STATIC_LU_PATH, dem_cached, target_has)

    for csv_path in csv_files:
        csv_name = os.path.basename(csv_path)
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
            os.makedirs(os.path.join(OUTPUT_VOL_DIR, ensemble, "nc", ha_num), exist_ok=True)

        if DRY_RUN:
            for dem_path in dem_files:
                ha_num = extract_ha_num(dem_path)
                for _, row in events_df.iterrows():
                    event_base = (
                        f"soil_{ha_num}_{int(row['event_num'])}_{int(row['start_year'])}_"
                        f"{int(row['start_month'])}_{int(row['start_day'])}_{ensemble}"
                    )
                    print(f"INFO {csv_name} | DEM {ha_num}")
                    print(os.path.join(OUTPUT_VOL_DIR, ensemble, "nc", ha_num, f"{event_base}_vol_30m.nc"))
            continue

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_event, row, dem_path, ensemble, porosity_grids, lu_grids, dem_cached, target_has
                )
                for dem_path in dem_files
                for _, row in events_df.iterrows()
            ]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()