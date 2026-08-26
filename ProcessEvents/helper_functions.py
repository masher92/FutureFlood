import glob
import os
import re
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
import matplotlib.pyplot as plt
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

# --------------------------
# Helper functions (copied from the main pipeline script)
# --------------------------
def extract_ha_num(path_or_name):
    """Extract an HA identifier such as 27, 27_a, or 54_d."""
    base = os.path.splitext(os.path.basename(path_or_name))[0]
    match = re.search(r"(\d+(?:_[A-Za-z0-9]+)?)$", base)
    return match.group(1) if match else None


def find_matching_netcdf(netcdf_dir, year, month, day):
    """Return the NetCDF whose date range contains the event date."""
    target_date_int = int(f"{year:04d}{month:02d}{day:02d}")
    for nc_file in glob.glob(os.path.join(netcdf_dir, "*.nc")):
        match = re.search(r"_(\d{8})-(\d{8})_", os.path.basename(nc_file))
        if match and int(match.group(1)) <= target_date_int <= int(match.group(2)):
            return nc_file
    print(f"[NetCDF Search Fail] No match for date {target_date_int} in {netcdf_dir}")
    return None


def clip_gravitational(ds, var, bounds, CRS_TARGET):
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
    """Resample clipped gravitational (or other) data to the DEM grid."""
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
    return resampled


def load_soil_grids(porosity_csv, soils_lookup_csv, soil_texture_dir, target_has):
    """Load porosity and effective depth (Lu) grids together, one soil-texture
    raster read per catchment, so both grids are guaranteed pixel-aligned.

    Lu[m] = 0.06375 * sqrt(Ks[cm/hr])  (Green-Ampt Lu = 4*sqrt(Ks), unit-converted)
    """
    porosity_df = pd.read_csv(porosity_csv)
    porosity_map = porosity_df.set_index("USDA_soil")["Effective_porosity"].to_dict()

    hc_df = pd.read_csv(soils_lookup_csv)
    hc_map = hc_df.set_index("USDA_soil")["Hydraulic_conductivity"].to_dict()

    soil_grids = {}
    for tif in glob.glob(os.path.join(soil_texture_dir, "*.tif")):
        ha_num = extract_ha_num(tif)
        if ha_num is None or ha_num not in target_has:
            continue

        with rasterio.open(tif) as src:
            data = src.read(1)
            valid = data != src.nodata

            porosity = np.vectorize(lambda v: porosity_map.get(v, np.nan))(data)
            porosity[~valid] = np.nan

            hc_values = np.vectorize(lambda v: hc_map.get(v, np.nan))(data)
            hc_values[~valid] = np.nan
            lu_values = 0.06375 * np.sqrt(hc_values)

            soil_grids[ha_num] = {"porosity": porosity, "lu": lu_values, 'hc':hc_values}

    return soil_grids

def aggregate_to_coarse_grid(fine_array, dem_info, coarse_da, resampling=Resampling.average):
    """Aggregate a 30m array back onto the coarser (e.g. 5km) grid that
    coarse_da is defined on, area-weighted-averaging over valid pixels only."""
    coarse_shape = np.squeeze(coarse_da.values).shape
    coarse_array = np.full(coarse_shape, np.nan, dtype=np.float32)

    fine = fine_array.astype(np.float32).copy()
    fine[np.isnan(fine)] = np.nan  # ensure masked areas are NaN, not e.g. -9999

    reproject(
        source=fine,
        destination=coarse_array,
        src_transform=dem_info["transform"],
        src_crs=dem_info["crs"],
        dst_transform=coarse_da.rio.transform(),
        dst_crs=coarse_da.rio.crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return coarse_array