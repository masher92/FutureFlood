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

def aggregate_to_5km(rainfall_5km_fp, raster, return_qa_array=False):
    """
    Aggregate 30m hydraulic conductivity values to 5km grid cells by taking
    the mean HK value of all valid 30m pixels within each 5km cell.
    Optionally writes a QA raster showing which 5km cell each 30m pixel was assigned to.
    """
    # Load a UKCP 5km NetCDF file purely to extract the 5km grid definition
    # (coordinates, size, spacing). The actual rainfall data inside is ignored —
    # it's just being used as a spatial reference grid.
    ds = xr.open_dataset(GRID_5KM_FILE)
    x5 = ds["projection_x_coordinate"]   # 1D array of 5km cell centre eastings
    y5 = ds["projection_y_coordinate"]   # 1D array of 5km cell centre northings
    nx = x5.size                          # number of columns in 5km grid
    ny = y5.size                          # number of rows in 5km grid
    dx = float(abs(x5[1] - x5[0]))       # cell spacing in x (should be ~5000m)
    dy = float(abs(y5[1] - y5[0]))       # cell spacing in y (should be ~5000m)
    # These are the outer edges of the 5km grid extent (cell centres ± half cell)
    xmin = float(x5.min() - dx/2)
    xmax = float(x5.max() + dx/2)
    ymin = float(y5.min() - dy/2)
    ymax = float(y5.max() + dy/2)

    # Reproject to British National Grid if not already in it
    if raster.rio.crs is None or raster.rio.crs.to_string() != "EPSG:27700":
        raster = raster.rio.reproject("EPSG:27700")

    # Boolean mask of non-NaN pixels
    valid = raster.notnull()

    # Get the 1D coordinate arrays for the 30m raster
    raster_x = raster.x.values
    raster_y = raster.y.values

    # Build a 2D grid of coordinates so every pixel has an (x, y) pair
    xx, yy = np.meshgrid(raster_x, raster_y)

    # Flatten everything to 1D for vectorised processing
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    raster_flat = raster.values.ravel()
    valid_flat = valid.values.ravel()

    # Keep ALL valid (non-NaN) pixels — unlike flood version we don't filter by > 0
    mask = valid_flat
    x_raster = x_flat[mask]
    y_raster = y_flat[mask]
    vals = raster_flat[mask]

    if x_raster.size == 0:
        return np.full((ny, nx), np.nan, dtype=np.float32)

    # Work out which 5km cell each 30m pixel falls into
    x5_arr = np.asarray(x5.values)
    y5_arr = np.asarray(y5.values)
    dx5 = float(abs(x5_arr[1] - x5_arr[0]))
    dy5 = float(abs(y5_arr[1] - y5_arr[0]))

    if x5_arr[1] > x5_arr[0]:
        xmin_edge = float(np.min(x5_arr) - dx5 / 2.0)
        ix = np.floor((x_raster - xmin_edge) / dx5).astype(np.int64)
    else:
        xmax_edge = float(np.max(x5_arr) + dx5 / 2.0)
        ix = np.floor((xmax_edge - x_raster) / dx5).astype(np.int64)

    if y5_arr[1] > y5_arr[0]:
        ymin_edge = float(np.min(y5_arr) - dy5 / 2.0)
        iy = np.floor((y_raster - ymin_edge) / dy5).astype(np.int64)
    else:
        ymax_edge = float(np.max(y5_arr) + dy5 / 2.0)
        iy = np.floor((ymax_edge - y_raster) / dy5).astype(np.int64)

    # Drop any pixels that fall outside the 5km grid extent
    in_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[in_grid]
    iy = iy[in_grid]
    vals = vals[in_grid]

    if ix.size == 0:
        return np.full((ny, nx), np.nan, dtype=np.float32)

    # Convert 2D (row, col) indices to a single 1D index for each pixel
    linear_idx = iy * nx + ix

    # QA block: write a 30m raster showing which 5km cell each pixel was assigned to
    # Useful for visually checking the mapping is correct — pixels should form
    # clean 5km blocks. Pass qa_out_tif="path/to/qa.tif" to enable.
    if return_qa_array is not None:
        # Indices into the flattened raster array for pixels that passed the mask
        raster_flat_idx = np.where(mask)[0]
        # Of those, only the ones that fell inside the 5km grid
        raster_flat_idx_in_grid = raster_flat_idx[in_grid]
        # Default to -1 (not assigned) for all pixels
        qa_flat = np.full(raster_flat.shape, -1, dtype=np.int32)
        # Fill in the 5km linear cell index for valid, in-grid pixels
        qa_flat[raster_flat_idx_in_grid] = linear_idx.astype(np.int32)
        qa_arr = qa_flat.reshape(raster.shape)
        qa_da = xr.DataArray(qa_arr, coords=raster.coords, dims=raster.dims)
#         qa_da = qa_da.rio.write_crs(raster.rio.crs)
#         qa_da.rio.write_transform(raster.rio.transform(), inplace=True)
#         os.makedirs(os.path.dirname(qa_out_tif), exist_ok=True)
#         qa_da.rio.to_raster(qa_out_tif)

    # Sum HK values per 5km cell, and count pixels per cell
    sum_flat   = np.bincount(linear_idx, weights=vals, minlength=nx * ny)
    count_flat = np.bincount(linear_idx, minlength=nx * ny)

    # Divide sum by count to get mean HK per cell; NaN where no pixels
    with np.errstate(invalid="ignore"):
        mean_hk = np.where(count_flat > 0, sum_flat / count_flat, np.nan)

    return mean_hk.reshape(ny, nx).astype(np.float32), qa_da

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

def filter_closer_to_catchment(cube, catchment_poly, plot=False, boundary_gdf = None):

    minx, miny, maxx, maxy = catchment_poly.bounds

    buffer = 2500  # metres

    constraint = iris.Constraint( projection_x_coordinate=lambda x: (minx - buffer) <= x <= (maxx + buffer),
        projection_y_coordinate=lambda y: (miny - buffer) <= y <= (maxy + buffer))

    sub_cube = cube.extract(constraint)
    
    if plot==True:
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': ccrs.OSGB()})
        qplt.pcolormesh(sub_cube, axes=ax, cmap = 'Blues', edgecolor='black', linewidth=0.5)
        boundary_gdf.boundary.plot(ax=ax, color='black');
        
    return sub_cube

#!/usr/bin/env python3

# -----------------------
# Paths (edit if needed)
# -----------------------
SOIL_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/Model_builds/Pluvial/v4/soil_texture/"
CATCHMENT_DIR = "/scratch/hydro4/users/kv25483/FutureFlood/Data/CatchmentShapefiles/"
GRID_5KM_FILE = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall/5km/Ens_01/bc_pr_rcp85_land-cpm_uk_5km_01_1hr_19901201-19911130.nc"

OUTPUT_DIR = "../../Data/HydraulicConductivity/"
SOIL_LOOKUP_FILE = "../../Data/Soil_Info.csv"


# -----------------------
# Helper: get all catchments
# -----------------------
def get_all_catchments(soil_dir):
    files = os.listdir(soil_dir)

    catchment_numbers = set()
    for f in files:
        match = re.search(r"soil_30m_(\d+)\.tif", f)
        if match:
            catchment_numbers.add(match.group(1))

    return sorted(catchment_numbers)


# -----------------------
# Main processing function
# -----------------------
def main(HA_NUM):

    print(f"\n--- Processing catchment {HA_NUM} ---")

    # -----------------
    # Lookup table
    # -----------------
    lookup_df = pd.read_csv(SOIL_LOOKUP_FILE)

    lookup = dict(zip(
        lookup_df["USDA_soil"],
        lookup_df["Hydraulic_conductivity"]))

    # -----------------
    # Catchment boundary
    # -----------------
    catchments = gpd.read_file(
        os.path.join(CATCHMENT_DIR, "hyd_areas_GB_with_subcatchments.shp"))

    this_catchment = catchments[catchments["HA_NUM"] == str(HA_NUM)]

    if this_catchment.empty:
        print(f"⚠️ Catchment {HA_NUM} not found — skipping")
        return

    # -----------------
    # Soil raster
    # -----------------
    soil_file = os.path.join(SOIL_DIR, f"soil_30m_{HA_NUM}.tif")

    if not os.path.exists(soil_file):
        print(f"⚠️ Missing file: {soil_file} — skipping")
        return

    soil_texture = rxr.open_rasterio(
        soil_file,
        chunks={"x": 2000, "y": 2000}).squeeze()

    soil_texture = soil_texture.where(soil_texture != -9999)

    # -----------------
    # Convert to hydraulic conductivity
    # -----------------
    soil_values = soil_texture.values
    hk_values = np.full(soil_values.shape, np.nan)

    for usda_class, hk in lookup.items():
        hk_values[soil_values == usda_class] = hk

    hk_raster = soil_texture.copy(data=hk_values)
    hk_raster.name = "hydraulic_conductivity"

    # -----------------
    # Aggregate to 5km
    # -----------------
    hk_5km, qa = aggregate_to_5km(
        GRID_5KM_FILE,
        hk_raster,
        return_qa_array=True)

    # -----------------
    # Convert to NetCDF
    # -----------------
    hk_out = convert_to_netcdf(GRID_5KM_FILE, hk_5km)

    # -----------------
    # Save
    # -----------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(OUTPUT_DIR, f"5km_{HA_NUM}.nc")

    hk_out.to_netcdf(output_file)

    print(f"✅ Saved: {output_file}")


# -----------------------
# CLI entry point
# -----------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert soil texture to hydraulic conductivity and aggregate to 5km")

    parser.add_argument(
        "--ha_num",
        type=str,
        help="Single catchment number (e.g. 23)")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all catchments"
    )

    args = parser.parse_args()

    if args.all:
        catchments = get_all_catchments(SOIL_DIR)
        print(f"Found {len(catchments)} catchments")

        for ha in catchments:
            main(ha)

    elif args.ha_num:
        main(args.ha_num)

    else:
        raise ValueError("Provide either --ha_num or --all")