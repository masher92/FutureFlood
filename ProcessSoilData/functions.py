import os
import re
import argparse
import matplotlib.pyplot as plt
import rioxarray as rxr
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import iris
import cartopy.crs as ccrs
import iris.quickplot as qplt

GRID_5KM_FILE = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall/5km/Ens_01/bc_pr_rcp85_land-cpm_uk_5km_01_1hr_19901201-19911130.nc"


def aggregate_to_5km(
    raster,
    agg_type="mean",  # "mean" or "mode"
    return_qa_array=False):
    """
    Aggregate 30m raster to 5km grid.

    Parameters
    ----------
    raster : xarray.DataArray
        Input 30m raster (must have x/y coords)
    agg_type : str
        "mean" (continuous) or "mode" (categorical)
    return_qa_array : bool
        If True, returns mapping of 30m pixels → 5km cell index

    Returns
    -------
    xarray.Dataset
        Aggregated result
    """

    # --- Load 5km grid definition ---
    ds = xr.open_dataset(GRID_5KM_FILE)
    x5 = ds["projection_x_coordinate"].values
    y5 = ds["projection_y_coordinate"].values
    nx, ny = len(x5), len(y5)

    # --- Get 30m coords ---
    x = raster["x"].values
    y = raster["y"].values
    xx, yy = np.meshgrid(x, y)

    # --- Flatten ---
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    vals = raster.values.ravel()

    # --- Remove NaNs ---
    valid = ~np.isnan(vals)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    vals = vals[valid]

    # --- Map to 5km indices ---
    ix = np.searchsorted(x5, x_flat) - 1
    iy = np.searchsorted(y5, y_flat) - 1

    # keep valid indices
    valid_idx = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[valid_idx]
    iy = iy[valid_idx]
    vals = vals[valid_idx]

    # --- Linear index ---
    linear_idx = iy * nx + ix
    n_cells = nx * ny
    print("Finished first part")

    # ============================================================
    # 🔹 MEAN aggregation (continuous)
    # ============================================================
    if agg_type == "mean":
        sums = np.bincount(linear_idx, weights=vals, minlength=n_cells)
        counts = np.bincount(linear_idx, minlength=n_cells)

        mean_vals = np.full(n_cells, np.nan)
        mask = counts > 0
        mean_vals[mask] = sums[mask] / counts[mask]

        out = mean_vals.reshape(ny, nx)

        ds_out = xr.Dataset({
            "aggregated_mean": (("projection_y_coordinate", "projection_x_coordinate"), out)})

    # ============================================================
    # 🔹 MODE aggregation (categorical)
    # ============================================================
    elif agg_type == "mode":
        vals = vals.astype(int)

        mode_vals = np.full(n_cells, np.nan)
        mode_prop = np.full(n_cells, np.nan)

        # group by cell
        for cell in tqdm(np.unique(linear_idx)):
            cell_vals = vals[linear_idx == cell]

            counts = np.bincount(cell_vals)
            mode = counts.argmax()
            prop = counts[mode] / counts.sum()

            mode_vals[cell] = mode
            mode_prop[cell] = prop

        ds_out = xr.Dataset({
            "mode": (("projection_y_coordinate", "projection_x_coordinate"),
                     mode_vals.reshape(ny, nx)),
            "mode_proportion": (("projection_y_coordinate", "projection_x_coordinate"),
                                mode_prop.reshape(ny, nx))})

    else:
        raise ValueError("agg_type must be 'mean' or 'mode'")

    # --- Assign coordinates ---
    ds_out = ds_out.assign_coords({"projection_x_coordinate": x5,
        "projection_y_coordinate": y5})

    # --- Optional QA output ---
    if return_qa_array:
        qa = np.full(xx.size, -1)
        qa[valid] = linear_idx
        qa = qa.reshape(xx.shape)

        return ds_out, qa

    return ds_out

def convert_to_netcdf(GRID_5KM_FILE, hk_array):

    ds = xr.open_dataset(GRID_5KM_FILE)
    x5 = ds["projection_x_coordinate"]
    y5 = ds["projection_y_coordinate"]

    # Ensure numpy array
    hk_array = np.asarray(hk_array)

    # Cell size
    dx5 = 5000.0
    dy5 = 5000.0

    # Bounds
    x_bounds = np.stack([x5.values - dx5/2, x5.values + dx5/2], axis=1)
    y_bounds = np.stack([y5.values - dy5/2, y5.values + dy5/2], axis=1)

    # Create DataArray
    hk_array_da = xr.DataArray(hk_array, dims=["projection_y_coordinate", "projection_x_coordinate"],
        coords={"projection_y_coordinate": y5.values,
            "projection_x_coordinate": x5.values}, name="hydraulic_conductivity")
    
    hk_array_da.coords["projection_x_coordinate"].attrs.update({"standard_name": "projection_x_coordinate","units": "m"})

    hk_array_da.coords["projection_y_coordinate"].attrs.update({"standard_name": "projection_y_coordinate","units": "m"})

    # Add bounds attribute
    hk_array_da.coords["projection_x_coordinate"].attrs["bounds"] = "projection_x_coordinate_bounds"
    hk_array_da.coords["projection_y_coordinate"].attrs["bounds"] = "projection_y_coordinate_bounds"

    # Metadata
    hk_array_da.attrs.update({
        "units": "m/day",
        "description": "Mean hydraulic conductivity aggregated from 30m to 5km",
        "grid_mapping": "crs"})

    # Convert to dataset
    hk_out = hk_array_da.to_dataset()

    # Add bounds variables
    hk_out["projection_x_coordinate_bounds"] = xr.DataArray(
        x_bounds, dims=["projection_x_coordinate", "bounds"])
    hk_out["projection_y_coordinate_bounds"] = xr.DataArray(
        y_bounds, dims=["projection_y_coordinate", "bounds"])

    # Add CRS variable (CF-compliant British National Grid)
    hk_out["crs"] = xr.DataArray( 0, attrs={
            "grid_mapping_name": "transverse_mercator",
            "latitude_of_projection_origin": 49.0,
            "longitude_of_central_meridian": -2.0,
            "false_easting": 400000.0,
            "false_northing": -100000.0,
            "scale_factor_at_central_meridian": 0.9996012717,
            "semi_major_axis": 6377563.396,
            "semi_minor_axis": 6356256.909,
            "longitude_of_prime_meridian": 0.0})
    return hk_out


