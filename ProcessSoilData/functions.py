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
from tqdm import tqdm

GRID_5KM_FILE = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall/5km/Ens_01/bc_pr_rcp85_land-cpm_uk_5km_01_1hr_19901201-19911130.nc"

def aggregate_to_5km(
    raster,
    agg_type="mean",
    return_qa_array=False,
    chunk_rows=500
):
    """
    Aggregate a large raster to the UKCP 5 km grid without loading
    the entire raster into memory.

    Spatial assignment:
        Each input 30 m pixel is assigned to the 5 km cell
        containing the CENTRE of that 30 m pixel.

    The 5 km cell boundaries are taken directly from:
        projection_x_coordinate_bnds
        projection_y_coordinate_bnds

    Parameters
    ----------
    raster : xarray.DataArray
        Input raster with x/y coordinates.

    agg_type : str
        "mean" or "mode".

    return_qa_array : bool
        If True, returns mapping of input pixels -> 5 km cell index.
        WARNING: this can itself be very large for high-resolution data.

    chunk_rows : int
        Number of raster rows to process at a time.

    Returns
    -------
    xarray.Dataset
        Aggregated result.

    Notes
    -----
    The 5 km coordinate variables are cell CENTRES, not cell
    boundaries. This was confirmed using the corresponding
    *_bnds variables in the UKCP NetCDF. For example:

        x = -197500
        x bounds = [-200000, -195000]

    so -197500 is exactly the centre of that 5 km cell.

    Therefore, aggregation assigns each 30 m pixel according to
    which 5 km cell contains its centre.
    """

    # ------------------------------------------------------------
    # Load 5 km grid
    # ------------------------------------------------------------

    ds = xr.open_dataset(GRID_5KM_FILE)

    x5 = ds["projection_x_coordinate"].values
    y5 = ds["projection_y_coordinate"].values

    # Actual 5 km cell boundaries
    x5_bnds = ds["projection_x_coordinate_bnds"].values
    y5_bnds = ds["projection_y_coordinate_bnds"].values

    # The first column contains the lower boundary and the
    # second column contains the upper boundary
    x_edges = x5_bnds[:, 0]
    x_edges = np.append(x_edges, x5_bnds[-1, 1])

    y_edges = y5_bnds[:, 0]
    y_edges = np.append(y_edges, y5_bnds[-1, 1])

    nx = len(x5)
    ny = len(y5)

    n_cells = nx * ny

    # ------------------------------------------------------------
    # Check that coordinate values really are cell centres
    # ------------------------------------------------------------

    x5_centres_from_bounds = (
        x5_bnds[:, 0] + x5_bnds[:, 1]
    ) / 2

    y5_centres_from_bounds = (
        y5_bnds[:, 0] + y5_bnds[:, 1]
    ) / 2

    if not np.allclose(x5, x5_centres_from_bounds):
        raise ValueError(
            "5 km x coordinates do not match the centres "
            "of their coordinate bounds."
        )

    if not np.allclose(y5, y5_centres_from_bounds):
        raise ValueError(
            "5 km y coordinates do not match the centres "
            "of their coordinate bounds."
        )

    # ------------------------------------------------------------
    # Input raster coordinates
    # ------------------------------------------------------------

    x = raster["x"].values
    y = raster["y"].values

    n_y = len(y)
    n_x = len(x)

    # ------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------

    if agg_type == "mean":

        sums = np.zeros(
            n_cells,
            dtype=np.float64
        )

        counts = np.zeros(
            n_cells,
            dtype=np.int64
        )

    elif agg_type == "mode":

        # Soil texture classes are 0-11.
        # Keep 12 available as well.
        max_class = 12

        class_counts = np.zeros(
            (max_class + 1, n_cells),
            dtype=np.int64
        )

    else:
        raise ValueError(
            "agg_type must be 'mean' or 'mode'"
        )

    # ------------------------------------------------------------
    # Optional QA array
    # ------------------------------------------------------------

    if return_qa_array:
        qa = np.full(
            (n_y, n_x),
            -1,
            dtype=np.int32
        )

    # ------------------------------------------------------------
    # Process raster in chunks
    # ------------------------------------------------------------

    for y_start in tqdm(
        range(0, n_y, chunk_rows),
        desc="Processing raster"
    ):

        y_end = min(
            y_start + chunk_rows,
            n_y
        )

        # Only load this chunk from the raster
        chunk = raster.isel(
            y=slice(y_start, y_end)
        )

        vals = chunk.values

        # Coordinates for this chunk
        y_chunk = y[y_start:y_end]

        # --------------------------------------------------------
        # Create coordinates for pixel CENTRES
        # --------------------------------------------------------

        x_flat = np.tile(
            x,
            len(y_chunk)
        )

        y_flat = np.repeat(
            y_chunk,
            n_x
        )

        vals_flat = vals.ravel()

        # --------------------------------------------------------
        # Map each 30 m pixel CENTRE to a 5 km cell
        # --------------------------------------------------------

        # IMPORTANT:
        # x_edges/y_edges are the actual 5 km CELL BOUNDARIES.
        #
        # Therefore this asks:
        # "Which 5 km cell contains the centre of this
        #  30 m pixel?"

        ix = np.searchsorted(
            x_edges,
            x_flat,
            side="right"
        ) - 1

        iy = np.searchsorted(
            y_edges,
            y_flat,
            side="right"
        ) - 1

        valid_idx = (
            (ix >= 0) &
            (ix < nx) &
            (iy >= 0) &
            (iy < ny)
        )

        # Remove pixels outside the 5 km grid
        ix = ix[valid_idx]
        iy = iy[valid_idx]
        vals_flat = vals_flat[valid_idx]

        # Remove NaNs
        valid_values = ~np.isnan(vals_flat)

        ix = ix[valid_values]
        iy = iy[valid_values]
        vals_flat = vals_flat[valid_values]

        # Linear 5 km cell index
        linear_idx = iy * nx + ix

        # --------------------------------------------------------
        # Mean
        # --------------------------------------------------------

        if agg_type == "mean":

            sums += np.bincount(
                linear_idx,
                weights=vals_flat,
                minlength=n_cells
            )

            counts += np.bincount(
                linear_idx,
                minlength=n_cells
            )

        # --------------------------------------------------------
        # Mode
        # --------------------------------------------------------

        elif agg_type == "mode":

            vals_int = vals_flat.astype(np.int16)

            for category in range(max_class + 1):

                category_mask = (
                    vals_int == category
                )

                if not np.any(category_mask):
                    continue

                category_cells = (
                    linear_idx[category_mask]
                )

                class_counts[category] += np.bincount(
                    category_cells,
                    minlength=n_cells
                )

        # --------------------------------------------------------
        # QA
        # --------------------------------------------------------

        if return_qa_array:

            # Mapping for ALL pixels in this chunk,
            # irrespective of whether their value is NaN.

            chunk_linear_idx = np.full(
                vals.size,
                -1,
                dtype=np.int32
            )

            xx_flat = np.tile(
                x,
                len(y_chunk)
            )

            yy_flat = np.repeat(
                y_chunk,
                n_x
            )

            ix_all = np.searchsorted(
                x_edges,
                xx_flat,
                side="right"
            ) - 1

            iy_all = np.searchsorted(
                y_edges,
                yy_flat,
                side="right"
            ) - 1

            valid_all = (
                (ix_all >= 0) &
                (ix_all < nx) &
                (iy_all >= 0) &
                (iy_all < ny)
            )

            chunk_linear_idx[valid_all] = (
                iy_all[valid_all] * nx
                + ix_all[valid_all]
            )

            qa[y_start:y_end, :] = (
                chunk_linear_idx.reshape(
                    len(y_chunk),
                    n_x
                )
            )

    # ============================================================
    # Construct output
    # ============================================================

    if agg_type == "mean":

        mean_vals = np.full(
            n_cells,
            np.nan,
            dtype=np.float64
        )

        mask = counts > 0

        mean_vals[mask] = (
            sums[mask] / counts[mask]
        )

        out = mean_vals.reshape(
            ny,
            nx
        )

        ds_out = xr.Dataset({
            "aggregated_mean": (
                (
                    "projection_y_coordinate",
                    "projection_x_coordinate"
                ),
                out
            )
        })

    elif agg_type == "mode":

        mode_vals = np.full(
            n_cells,
            np.nan,
            dtype=np.float64
        )

        mode_prop = np.full(
            n_cells,
            np.nan,
            dtype=np.float64
        )

        total_counts = (
            class_counts.sum(axis=0)
        )

        valid_cells = (
            total_counts > 0
        )

        mode_indices = np.argmax(
            class_counts,
            axis=0
        )

        mode_vals[valid_cells] = (
            mode_indices[valid_cells]
        )

        mode_counts = (
            class_counts[
                mode_indices,
                np.arange(n_cells)
            ]
        )

        mode_prop[valid_cells] = (
            mode_counts[valid_cells]
            / total_counts[valid_cells]
        )

        ds_out = xr.Dataset({

            "mode": (
                (
                    "projection_y_coordinate",
                    "projection_x_coordinate"
                ),
                mode_vals.reshape(ny, nx)
            ),

            "mode_proportion": (
                (
                    "projection_y_coordinate",
                    "projection_x_coordinate"
                ),
                mode_prop.reshape(ny, nx)
            )
        })

    # ------------------------------------------------------------
    # Assign coordinates
    # ------------------------------------------------------------

    ds_out = ds_out.assign_coords({
        "projection_x_coordinate": x5,
        "projection_y_coordinate": y5
    })

    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------

    if return_qa_array:
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


