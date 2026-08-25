#!/usr/bin/env python3
"""
Second-stage post-processing: reads existing vol_sm GeoTIFFs (from an
earlier run of the tif-based pipeline) and derives IMD and remaining
upper-zone capacity, without re-touching any NetCDF/resampling logic for
the raw soil moisture data.

Writes a single combined netCDF per event -- same {vol_sm, imd,
remaining_capacity} format as soil_pipeline.py's production output -- so
downstream code can treat outputs from both scripts identically regardless
of which one produced them.

Expects the existing directory structure:
    <VOL_SM_ROOT>/<ensemble>/tif/<HA_NUM>/soil_<HA_NUM>_<event>_<y>_<m>_<d>_<ensemble>_vol_30m.tif

Writes:
    <OUTPUT_NC_DIR>/<ensemble>/nc/<HA_NUM>/..._vol_30m.nc
"""
import argparse
import glob
import logging
import os
from datetime import datetime
from tqdm import tqdm

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from shapely.geometry import box

from soil_pipeline import (CRS_TARGET,POROSITY_CSV,SOIL_TEXTURE_DIR,STATIC_LU_PATH,array_to_dataarray,
                           load_porosity_grids,resample_to_30m_array,)
from functions import aggregate_to_5km

VOL_SM_ROOT = ("/scratch/hydro5/users/la17355/FUTURE-FLOOD/UKCP_soil_moisture/volumetric_30m/soil_depth_missing_tests/")

# NOTE: matches the production pipeline's OUTPUT_VOL_DIR location
# (kv25483's scratch space) -- double check this is deliberate.
OUTPUT_NC_DIR = "/scratch/hydro4/users/kv25483/FutureFlood/Data/SoilDynamicVariables/combined_from_tif/"

LOG_DIR = "/scratch/hydro4/users/kv25483/FutureFlood"


def build_dem_info_from_tif(tif_path):
    """Grid info read directly from the vol_sm tif itself, so IMD/capacity
    are guaranteed to land on the exact same grid vol_sm was written on --
    no risk of drift if resampling parameters ever change upstream."""
    with rasterio.open(tif_path) as src:
        return {"transform": src.transform,"crs": src.crs,"width": src.width,"height": src.height,"bounds": src.bounds,}


def load_static_grids_for_ha(ha_num, dem_info):
    """Porosity + Lu, both clipped and resampled onto this exact grid.

    Porosity is resampled here rather than used at its native resolution,
    because the native soil-texture raster and the DEM-derived vol_sm grid
    are two independently-built products with no guaranteed pixel-for-pixel
    alignment -- confirmed earlier by their bounding boxes not overlapping
    at all for a mismatched HA_NUM. Always resample onto dem_info, never
    assume native alignment.
    """
    porosity_grids = load_porosity_grids(POROSITY_CSV, SOIL_TEXTURE_DIR, {ha_num})
    if ha_num not in porosity_grids:
        return None, None

    entry = porosity_grids[ha_num]
    porosity_da = ( xr.DataArray(entry["grid"], dims=("y", "x"))
        .rio.write_crs(entry["crs"])
        .rio.write_transform(entry["transform"]))
    porosity_data, _ = resample_to_30m_array(porosity_da, dem_info)

    lu_national = rioxarray.open_rasterio(STATIC_LU_PATH, masked=True).squeeze()
    if not lu_national.rio.crs:
        lu_national = lu_national.rio.write_crs(CRS_TARGET)
    bounds_geom = box(*dem_info["bounds"])
    lu_clipped = lu_national.rio.clip_box(*bounds_geom.bounds)
    lu_30m, _ = resample_to_30m_array(lu_clipped, dem_info)

    return porosity_data, lu_30m

def compute_imd_and_capacity(vol_sm_path,porosity_data,lu_data):

    with rasterio.open(vol_sm_path) as src:

        vol_sm = src.read(1).astype(np.float32)

        transform = src.transform
        crs = src.crs

        nodata = ( src.nodata if src.nodata is not None else -9999)

    # ------------------------------------------------------------
    # Valid pixels
    # ------------------------------------------------------------

    valid = ((vol_sm != nodata)& np.isfinite(vol_sm)& np.isfinite(porosity_data)& np.isfinite(lu_data))

    # Convert vol_sm nodata to NaN
    vol_sm = np.where(valid,vol_sm,np.nan)

    # ------------------------------------------------------------
    # Calculate IMD and remaining capacity
    # ------------------------------------------------------------

    with np.errstate(
        divide="ignore",
        invalid="ignore"):

        imd = (porosity_data* (1.0 - vol_sm))

        remaining_capacity = (imd* lu_data)

    # Explicitly enforce the mask
    imd[~valid] = np.nan
    remaining_capacity[~valid] = np.nan

    # Round
    imd = np.round(imd, 4).astype(np.float32)

    remaining_capacity = np.round(remaining_capacity, 4).astype(np.float32)

    return (vol_sm,imd,remaining_capacity,transform,crs,np.nan)

def aggregate_array_to_5km(array,transform,crs,agg_type="mean"):
    """
    Convert a 30 m numpy array to a DataArray using its raster
    transform, then aggregate to the UKCP 5 km grid using
    centre-based pixel assignment.
    """

    height, width = array.shape

    # Build x/y coordinates from the raster transform
    x = ( transform.c + (np.arange(width) + 0.5) * transform.a)

    y = (transform.f+ (np.arange(height) + 0.5) * transform.e)

    da = xr.DataArray(array,dims=("y", "x"),coords={    "y": y,    "x": x})

    # Aggregate using the function we just updated
    result = aggregate_to_5km(da,agg_type=agg_type)

    return result

def write_combined_nc(vol_sm, imd, remaining_capacity, transform, crs, nodata, nc_out):
    """Write vol_sm, imd, and remaining_capacity as three variables in a
    single combined netCDF, matching soil_pipeline.py's output format."""
    ds = xr.Dataset({"vol_sm": array_to_dataarray(vol_sm, transform, crs, "vol_sm", nodata),
            "imd": array_to_dataarray(imd, transform, crs, "imd", nodata),
            "remaining_capacity": array_to_dataarray(remaining_capacity, transform, crs, "remaining_capacity", nodata), })
    ds["vol_sm"].attrs["units"] = "1"
    ds["imd"].attrs["units"] = "1"
    ds["remaining_capacity"].attrs["units"] = "m"

    os.makedirs(os.path.dirname(nc_out), exist_ok=True)
    ds.to_netcdf(nc_out)


# def derive_and_write(vol_sm_path, porosity_data, lu_data, nc_out):
#     """Combined compute + write, for use inside the batch loop below."""
#     vol_sm, imd, remaining_capacity, transform, crs, nodata = compute_imd_and_capacity(
#         vol_sm_path, porosity_data, lu_data)
#     write_combined_nc(vol_sm, imd, remaining_capacity, transform, crs, nodata, nc_out)

def derive_and_write(vol_sm_path,porosity_data,lu_data,nc_out):

    # ------------------------------------------------------------
    # Calculate everything at 30 m
    # ------------------------------------------------------------

    vol_sm, imd, remaining_capacity, transform, crs, nodata = (compute_imd_and_capacity(vol_sm_path,porosity_data,lu_data))

#     print("vol_sm:")
#     print("  min:", np.nanmin(vol_sm))
#     print("  max:", np.nanmax(vol_sm))
#     print("  NaNs:", np.isnan(vol_sm).sum())

#     print("imd:")
#     print("  min:", np.nanmin(imd))
#     print("  max:", np.nanmax(imd))
#     print("  NaNs:", np.isnan(imd).sum())

#     print("capacity:")
#     print("  min:", np.nanmin(remaining_capacity))
#     print("  max:", np.nanmax(remaining_capacity))
#     print("  NaNs:", np.isnan(remaining_capacity).sum())

    
    # ------------------------------------------------------------
    # Aggregate the resulting 30 m variables to 5 km
    # ------------------------------------------------------------

    vol_sm_5km = aggregate_array_to_5km(vol_sm,transform,crs,agg_type="mean")
    imd_5km = aggregate_array_to_5km(imd,transform,crs,agg_type="mean")
    capacity_5km = aggregate_array_to_5km(remaining_capacity,transform,crs,agg_type="mean")

    # ------------------------------------------------------------
    # Rename variables
    # ------------------------------------------------------------

    ds_out = xr.Dataset({"vol_sm": vol_sm_5km["aggregated_mean"],
        "imd": imd_5km["aggregated_mean"],"remaining_capacity": capacity_5km["aggregated_mean"]})

    ds_out["vol_sm"].attrs["units"] = "1"
    ds_out["imd"].attrs["units"] = "1"
    ds_out["remaining_capacity"].attrs["units"] = "m"
    
    #print(ds_out["imd"].values)
    
    # ------------------------------------------------------------
    # Save ONLY the 5 km result
    # ------------------------------------------------------------

    os.makedirs(os.path.dirname(nc_out),exist_ok=True)
    
    # ------------------------------------------------------------
    # Crop output to the area containing valid data
    # ------------------------------------------------------------

    valid = (np.isfinite(ds_out["vol_sm"]) |np.isfinite(ds_out["imd"]) |np.isfinite(ds_out["remaining_capacity"]))

    # Find the 5 km rows/columns containing any valid data
    valid_y = valid.any(dim="projection_x_coordinate")
    valid_x = valid.any(dim="projection_y_coordinate")

    ds_out = ds_out.sel(projection_y_coordinate=valid_y,projection_x_coordinate=valid_x)
    
    ds_out.to_netcdf(nc_out, engine="netcdf4")
    # return ds_out


def run_second_stage(target_has=None):
    """Scan VOL_SM_ROOT and derive IMD/capacity for every matching tif.
    If target_has is given (a set of HA_NUM strings), only process those.
    """

    static_cache = {}  # ha_num -> (porosity_data, lu_data)

    pattern = os.path.join(VOL_SM_ROOT, "*", "tif", "*", "*_vol_30m.tif")
    for vol_sm_path in tqdm(glob.glob(pattern)):
        parts = vol_sm_path.split(os.sep)
        ha_num = parts[-2]
        ensemble = parts[-4]

        fname = os.path.basename(vol_sm_path)
        event_base = fname.replace("_vol_30m.tif", "")
        # nc_out = os.path.join(OUTPUT_NC_DIR, ensemble, "nc", ha_num, f"{event_base}_vol_30m.nc")
        nc_out = os.path.join(OUTPUT_NC_DIR,ensemble,"nc",ha_num,f"{event_base}_vol_5km.nc")

        dem_info = build_dem_info_from_tif(vol_sm_path)

        if ha_num not in static_cache:
            porosity_data, lu_data = load_static_grids_for_ha(ha_num, dem_info)
            if porosity_data is None:
                logging.error("[Porosity Missing] %s", ha_num)
                continue
            static_cache[ha_num] = (porosity_data, lu_data)

        porosity_data, lu_data = static_cache[ha_num]
        derive_and_write(vol_sm_path, porosity_data, lu_data, nc_out)
        
run_second_stage("40")