import numpy as np
from shapely import contains_xy
import iris
from iris.cube import CubeList
import matplotlib.pyplot as plt
from shapely.geometry import box, MultiPolygon
from shapely.ops import unary_union
import cartopy.crs as ccrs
import iris.quickplot as qplt
import os
import re
import pandas as pd
import glob
import rasterio
from rasterio.warp import Resampling, reproject

from iris.warnings import IrisCfMissingVarWarning
import warnings
warnings.filterwarnings("ignore", category=IrisCfMissingVarWarning)
warnings.filterwarnings("ignore", message=".*ensemble_member_id.*")
iris.FUTURE.date_microseconds = True
iris.FUTURE.datum_support = True

def prepare_flood_cube(cube):
    """Apply standard transformations to a flood cube."""
    cube = set_osgb_projection(cube)
    cube = guess_bounds_if_missing(cube)
    # cube = filter_closer_to_catchment(cube, catchment_poly, plot=plot)
    return cube

def set_osgb_projection(cube):
    """Assign OSGB coordinate system to x/y coordinates."""
    bng = iris.coord_systems.OSGB()
    cube.coord('projection_x_coordinate').coord_system = bng
    cube.coord('projection_y_coordinate').coord_system = bng
    return cube

def guess_bounds_if_missing(cube):
    """Guess bounds on x/y coordinates if not already present."""
    for coord_name in ['projection_x_coordinate', 'projection_y_coordinate']:
        coord = cube.coord(coord_name)
        if not coord.has_bounds():
            coord.guess_bounds()
    return cube

def load_3d_cube(filepath):
    cubes = iris.load(filepath)
    matches = [c for c in cubes if c.ndim == 3]
    if len(matches) == 0:
        raise ValueError(f"No 3D cube found in {filepath}")
    if len(matches) > 1:
        raise ValueError(f"Multiple 3D cubes found in {filepath} — be more specific")
    return matches[0]

def filter_closer_to_catchment(cube, catchment_poly, plot=False, boundary_gdf = None):

    minx, miny, maxx, maxy = catchment_poly.bounds

    buffer = 0  # metres

    constraint = iris.Constraint( projection_x_coordinate=lambda x: (minx - buffer) <= x <= (maxx + buffer),
        projection_y_coordinate=lambda y: (miny - buffer) <= y <= (maxy + buffer))

    sub_cube = cube.extract(constraint)
    
    if plot==True:
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'projection': ccrs.OSGB()})
        qplt.pcolormesh(sub_cube[2], axes=ax, cmap = 'terrain', edgecolor='black', linewidth=0.5)
        ax.add_geometries([catchment_poly], crs=ccrs.OSGB(), facecolor='none', edgecolor='black', linewidth=1.5)
        
    return sub_cube


def mask_cube_with_catchment_full_grid(cube, catchment_poly, method="full_cell"):
    """
    Build a 2D boolean mask on the full grid (e.g. full UK grid).
    Returns (ny, nx) boolean array — True where inside catchment.
    Only needs to be called once per catchment.
    """
    x = cube.coord('projection_x_coordinate').points
    y = cube.coord('projection_y_coordinate').points
    xx, yy = np.meshgrid(x, y)

    if method == "center_point":
        mask = contains_xy(catchment_poly, xx.ravel(), yy.ravel())
        mask = mask.reshape(xx.shape)

    elif method == "full_cell":
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        cells = [box(xi - dx/2, yi - dy/2, xi + dx/2, yi + dy/2)
            for yi, xi in zip(yy.ravel(), xx.ravel())]

        tree = STRtree(cells)
        hits = tree.query(catchment_poly, predicate='intersects')

        mask = np.zeros(len(cells), dtype=bool)
        mask[hits] = True
        mask = mask.reshape(xx.shape)

    else:
        raise ValueError("method must be 'center_point' or 'full_cell'")

    return mask  # (ny, nx) boolean, never applied to any cube


DATASET_MIN_YEAR = 1990
def get_rainfall_cube_subsection(yr, ENS_NUM, RAINFALLDIR, start_idx=None, stop_idx=None):
    HOURS_PER_MONTH = 30 * 24

    all_files  = []
    cumulative = 0

    # File 0: December of the previous calendar year — only exists if yr-1 is in range
    has_prev_december = not (yr == DATASET_MIN_YEAR)
    if has_prev_december:
        all_files.append({
            'path':         f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr-1}1201-{yr-1}1230.nc",
            'global_start': cumulative,
            'global_end':   cumulative + HOURS_PER_MONTH})
        cumulative += HOURS_PER_MONTH

    for m in range(1, 12):
        all_files.append({
            'path':         f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr}{m:02d}01-{yr}{m:02d}30.nc",
            'global_start': cumulative,
            'global_end':   cumulative + HOURS_PER_MONTH})
        cumulative += HOURS_PER_MONTH

    all_files.append({
        'path':         f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr}1201-{yr}1230.nc",
        'global_start': cumulative,
        'global_end':   cumulative + HOURS_PER_MONTH})

    if start_idx is not None and stop_idx is not None:
        if not has_prev_december and start_idx < HOURS_PER_MONTH:
            raise ValueError(
                f"Event requests start_idx={start_idx} (before 1 Jan {yr}), but no "
                f"December {yr-1} file exists — dataset begins {DATASET_MIN_YEAR}1201. "
                f"This event cannot be loaded.")
        needed = [f for f in all_files
                  if f['global_end'] > start_idx and f['global_start'] < stop_idx]
    else:
        needed = all_files

    monthly_cubes = iris.cube.CubeList()
    for f in needed:
        cube = iris.load(f['path'])[1]
        cube.attributes = {}
        monthly_cubes.append(cube)

    for cube in monthly_cubes:
        cube.coord('time').bounds = None

    year_cube = monthly_cubes.concatenate_cube()

    if start_idx is not None and stop_idx is not None:
        offset = needed[0]['global_start']
        return year_cube[start_idx - offset:stop_idx - offset, :, :]

    return year_cube


def subset_cube_to_bbox(cube, catchment_poly, buffer=0):
    minx, miny, maxx, maxy = catchment_poly.bounds

    x_full = cube.coord('projection_x_coordinate').points
    y_full = cube.coord('projection_y_coordinate').points

    constraint = iris.Constraint(
        projection_x_coordinate=lambda x: (minx - buffer) <= x <= (maxx + buffer),
        projection_y_coordinate=lambda y: (miny - buffer) <= y <= (maxy + buffer))

    sub_cube = cube.extract(constraint)

    # Record where the sub-cube starts in the full grid
    x_offset = int(np.searchsorted(x_full, sub_cube.coord('projection_x_coordinate').points[0]))
    y_offset = int(np.searchsorted(y_full, sub_cube.coord('projection_y_coordinate').points[0]))

    return sub_cube, x_offset, y_offset

def plot_cluster_check(rain, peak_slice, threshold_mask, cluster_mask,
                       row, y0, y1, x0, x1,
                       threshold_level, neighbourhood_sum, neighbourhood_size,
                       flood=None, catchment_poly=None, boundary_gdf=None,
                       buffer=2000):
    """
    Diagnostic plot to verify cluster extraction, using the same
    iris/cartopy plotting approach as plot_peak_event.
    
    Panels:
      1. Threshold mask — all cells >= threshold_value, outlined as polygons
      2. Cluster mask   — only the connected cluster containing the peak cell
      3. Flood extent   — optional, with cluster outline overlaid
    
    Parameters
    ----------
    rain             : 2D numpy array — cleaned rainfall field (NaNs for invalid)
    peak_slice       : iris Cube      — 2D spatial slice at peak timestep (for qplt)
    threshold_mask   : 2D bool array  — cells >= threshold_value
    cluster_mask     : 2D bool array  — cells in the peak-containing cluster only
    row              : namedtuple     — event row from itertuples()
    y0,y1,x0,x1     : int            — neighbourhood bounding box indices
    threshold_level  : float          — e.g. 0.8, used in title
    neighbourhood_sum: float          — precomputed neighbourhood sum, shown in title
    flood            : iris Cube      — optional 2D flood extent cube
    catchment_poly   : shapely geom   — catchment boundary for zooming
    boundary_gdf     : GeoDataFrame   — catchment boundary for plotting
    buffer           : int            — metres of padding around catchment extent
    """

    # ── Get coordinate arrays for polygon conversion ──────────────────────────
    # These are the actual projected coordinates (metres, OSGB) of each grid cell,
    # needed by mask_to_polygons to create properly georeferenced shapely polygons
    x = peak_slice.coord('projection_x_coordinate').points
    y = peak_slice.coord('projection_y_coordinate').points

    # ── Convert boolean masks to shapely polygons ─────────────────────────────
    # mask_to_polygons traces the boundary of each connected True region and
    # returns a list of shapely Polygons in OSGB coordinates — same as plot_peak_event
    threshold_polys = mask_to_polygons(threshold_mask, x, y)
    cluster_polys   = mask_to_polygons(cluster_mask,   x, y)

    # ── Also build a polygon for the neighbourhood bounding box ──────────────
    # Converts the array index bounds (y0,y1,x0,x1) into coordinate space so
    # the box is drawn correctly on the map rather than in index space
    dx          = x[1] - x[0]
    dy          = y[1] - y[0]
    box_x0      = x[x0] - dx / 2
    box_x1      = x[x1 - 1] + dx / 2
    box_y0      = y[y0] - dy / 2
    box_y1      = y[y1 - 1] + dy / 2
    neighbourhood_box = box(box_x0, box_y0, box_x1, box_y1)  # shapely.geometry.box

    # ── Layout ────────────────────────────────────────────────────────────────
    ncols = 3 if flood is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), subplot_kw={'projection': ccrs.OSGB()})
    axes = list(axes)

    panels = [("Threshold mask\n"
         f"(>= {threshold_level*100:.0f}% of peak)",  threshold_polys, axes[0]),
        ("Peak cluster\n"
         f"neighbourhood sum = {neighbourhood_sum:.2f} mm", cluster_polys, axes[1]),]

    # ── Rainfall panels ───────────────────────────────────────────────────────
    for title, polys, ax in panels:
        # Use qplt so iris handles the coordinate-aware pcolormesh
        qplt.pcolormesh(peak_slice, axes=ax, cmap='Blues')

        # Draw the mask outline polygons (threshold or cluster)
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color='orange', linewidth=2, transform=ccrs.OSGB())

        # Mark the peak cell
        ax.scatter(row['rainpeak_x_coord'], row['rainpeak_y_coord'], s=50, color='red', zorder=6,
                   transform=ccrs.OSGB(),
                   label=f"Peak ({int(row['rainpeak_x_global'])},{int(row['rainpeak_y_global'])})")

        # Draw the neighbourhood bounding box
        bx, by = neighbourhood_box.exterior.xy
        ax.plot(bx, by, color='red', linewidth=1.5, linestyle='--',
                transform=ccrs.OSGB(), label=f'Neighbourhood (n={neighbourhood_size})')

        ax.set_title(title)
        ax.legend(fontsize=7, loc='lower right')

    # ── Optional flood panel ──────────────────────────────────────────────────
    if flood is not None:
        ax = axes[2]
        qplt.pcolormesh(flood, axes=ax, cmap='Blues')
        ax.set_title("Flood extent")
        ax.scatter(row['rainpeak_x_coord'], row['rainpeak_y_coord'], s=50, color='red',
                   zorder=6, transform=ccrs.OSGB())

        # Overlay the cluster outline on the flood panel for comparison
        for i, poly in enumerate(cluster_polys):
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color='green', linewidth=2, transform=ccrs.OSGB(),
                    label='Cluster' if i == 0 else None)
        ax.legend(fontsize=8)

    # ── Shared formatting ─────────────────────────────────────────────────────
    for ax in axes:
        if catchment_poly is not None:
            minx, miny, maxx, maxy = catchment_poly.bounds
            ax.set_extent(
                [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer],
                crs=ccrs.OSGB())
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color='black')

    plt.suptitle(
        f"Cluster extraction check — ens={row['ens']}, year={row['start_year']}, event={row['event_num']}\n"
        f"Peak = {row['rainpeak_peak_value']:.2f} mm, threshold = {threshold_level*100:.0f}%"
        f" ({row['rainpeak_peak_value'] * threshold_level:.2f} mm)",
        fontsize=11)
    plt.tight_layout()
    plt.show()
    
def mask_to_polygons(mask, x, y):
    dx, dy = x[1] - x[0], y[1] - y[0]
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []
    cells = [
        box(x[xi]-dx/2, y[yi]-dy/2, x[xi]+dx/2, y[yi]+dy/2)
        for yi, xi in zip(ys, xs)
    ]
    # buffer(0) is faster than unary_union for simple grids
    shape = unary_union(cells).buffer(0)
    return list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]        


# --------------------------
# Helper functions (copied from the main pipeline script)
# --------------------------
def extract_ha_num(path_or_name):
    """Extract an HA identifier such as 27, 27_a, or 54_d."""
    base = os.path.splitext(os.path.basename(path_or_name))[0]
    match = re.search(r"(\d+(?:_[A-Za-z0-9]+)?)$", base)
    return match.group(1) if match else None

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

            soil_grids[ha_num] = {"texture" : valid, "porosity": porosity, "lu": lu_values, 'hc':hc_values}

    return soil_grids

def filter_closer_to_catchment_xr_bounds(da_or_ds, catchment_poly, x_dim="projection_x_coordinate",
                                           y_dim="projection_y_coordinate", cell_size=5000, buffer=0):
    minx, miny, maxx, maxy = catchment_poly.bounds
    x_full = da_or_ds[x_dim].values
    y_full = da_or_ds[y_dim].values
    half = cell_size / 2

    x_mask = (x_full + half >= minx - buffer) & (x_full - half <= maxx + buffer)
    y_mask = (y_full + half >= miny - buffer) & (y_full - half <= maxy + buffer)

    return da_or_ds.isel({x_dim: np.where(x_mask)[0], y_dim: np.where(y_mask)[0]})

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
        resampling=resampling,)
    return coarse_array

def find_matching_netcdf(netcdf_dir, year, month, day):
    """Return the NetCDF whose date range contains the event date."""
    target_date_int = int(f"{year:04d}{month:02d}{day:02d}")
    for nc_file in glob.glob(os.path.join(netcdf_dir, "*.nc")):
        match = re.search(r"_(\d{8})-(\d{8})_", os.path.basename(nc_file))
        if match and int(match.group(1)) <= target_date_int <= int(match.group(2)):
            return nc_file
    print(f"[NetCDF Search Fail] No match for date {target_date_int} in {netcdf_dir}")
    return None

def resample_to_30m_array(source_data, dem):
    """Resample clipped gravitational (or other) data to the DEM grid."""
    source_array = np.squeeze(source_data.values)
    if source_array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {source_array.shape}")

    resampled = np.empty((dem["height"], dem["width"]), dtype=source_array.dtype)
    reproject(source=source_array,
        destination=resampled,
        src_transform=source_data.rio.transform(),
        src_crs=source_data.rio.crs,
        dst_transform=dem["transform"],
        dst_crs=dem["crs"],
        resampling=Resampling.nearest,)
    return resampled