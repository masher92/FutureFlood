import warnings
import numpy as np
import pandas as pd
import iris
from iris.cube import CubeList
import geopandas as gpd
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from iris.warnings import IrisCfMissingVarWarning
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import iris.plot as iplt
import cftime
from shapely.ops import unary_union
from shapely.geometry import box, MultiPolygon
from shapely import contains_xy
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", category=IrisCfMissingVarWarning)
warnings.filterwarnings("ignore", message=".*ensemble_member_id.*")
iris.FUTURE.date_microseconds = True
iris.FUTURE.datum_support = True

def get_data_at_peak_cell(cube, one_event_results, x_idx_variable, y_idx_variable):

    x_idx = one_event_results[x_idx_variable]
    y_idx = one_event_results[y_idx_variable]

    x_coord =  one_event_results['x_coord']
    y_coord =  one_event_results['y_coord']

    cube_x = cube.coord('projection_x_coordinate').points
    cube_y = cube.coord('projection_y_coordinate').points

    # Check the coordinate at the global index matches what we stored
    assert np.isclose(cube_x[x_idx], x_coord, atol=100), \
        f"x mismatch: {cube_x[x_idx]} vs {x_coord}"
    assert np.isclose(cube_y[y_idx], y_coord, atol=100), \
        f"y mismatch: {cube_y[y_idx]} vs {y_coord}"

    # If those pass, extract the timeseries at that cell
    if len(cube.shape) == 2:
        cube_at_peak = cube[y_idx, x_idx]
    elif len(cube.shape) == 3:
        cube_at_peak = cube[:, y_idx, x_idx]
    return cube_at_peak  



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


def filter_closer_to_catchment_xr_bounds(da_or_ds, catchment_poly, x_dim="projection_x_coordinate",
                                           y_dim="projection_y_coordinate", cell_size=5000, buffer=0):
    minx, miny, maxx, maxy = catchment_poly.bounds
    x_full = da_or_ds[x_dim].values
    y_full = da_or_ds[y_dim].values
    half = cell_size / 2

    x_mask = (x_full + half >= minx - buffer) & (x_full - half <= maxx + buffer)
    y_mask = (y_full + half >= miny - buffer) & (y_full - half <= maxy + buffer)

    return da_or_ds.isel({x_dim: np.where(x_mask)[0], y_dim: np.where(y_mask)[0]})

def get_rainfall_cube(yr, ENS_NUM, RAINFALLDIR):
    """Load and concatenate monthly rainfall cubes for the event year.
    Loads Dec of previous year + Jan-Nov of target year.
    """
    files = [
        f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr-1}1201-{yr-1}1230.nc"]
    for m in range(1, 12):
        files.append(
            f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr}{m:02d}01-{yr}{m:02d}30.nc")

    monthly_cubes = iris.cube.CubeList()
    for f in files:
        cube = iris.load(f)[1]
        cube.attributes = {}
        monthly_cubes.append(cube)

    return monthly_cubes.concatenate_cube()




def find_temporal_profile(cube, details, plot):
    # Extract event time series
    values = cube.data[details['start_idx']:details['stop_idx']]
    times = cube.coord('yyyymmddhh').points[details['start_idx']:details['stop_idx']].astype(int)

    metrics = compute_temporal_metrics(values)
    
    # Convert to datetime
    dt = pd.to_datetime(times.astype(str), format="%Y%m%d%H")
    
    #temp_profile_dict = {'total_acc': values.sum(), 'times':dt, 'values':values}
    temp_profile_dict = {**metrics,'times': dt,'values': values}
    
    # Plot
    if plot == True:
        plt.figure(figsize=(8,4))
        plt.plot(dt, values, color='black')

        # Mark peak timestep
        rainfall_peak_time = cube.coord('yyyymmddhh').points[details['t_global']]
        peak_dt = pd.to_datetime(str(int(rainfall_peak_time)), format="%Y%m%d%H")
        plt.axvline(peak_dt, linestyle='--', color = 'red', label='Peak timestep')

        # Format ticks
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H'))

        plt.xlabel("Date")
        plt.ylabel("Precipitation intensity (mm/hr)")
        plt.title("Rainfall at peak grid cell during event")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return temp_profile_dict

def compute_temporal_metrics(values):

    values = np.asarray(values).flatten()
    n = len(values)

    # --- Basic ---
    total_acc = np.nansum(values)

    # Peak position ratio
    peak_idx = np.argmax(values)
    peak_position_ratio = peak_idx / max(n - 1, 1)

    # --- D50 ---
    if total_acc > 0:
        cum = np.cumsum(values) / total_acc
        time_pct = np.linspace(0, 100, n)

        idx = np.where(cum >= 0.5)[0]
        if len(idx) > 0:
            i = idx[0]
            if i == 0:
                d50 = time_pct[0]
            else:
                x1, y1 = time_pct[i-1], cum[i-1]
                x2, y2 = time_pct[i],   cum[i]
                d50 = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else np.nan
        else:
            d50 = np.nan
    else:
        d50 = np.nan

    # --- Gini coefficient (inequality of rainfall distribution) ---
    if n == 0 or np.all(values == 0):
        gini = 0.0
    else:
        s = np.sort(values)
        idx = np.arange(1, n + 1)
        gini = (2 * np.sum(idx * s) / (n * np.sum(s))) - (n + 1) / n

    # --- Time-based std (temporal spread around centre of mass) ---
    if total_acc > 0:
        pos = np.linspace(0, 1, n)  # normalised time
        t_cg = np.sum(pos * values) / total_acc
        time_based_std = np.sqrt(np.sum(((pos - t_cg) ** 2) * values) / total_acc)
    else:
        time_based_std = np.nan

    # --- Event dry ratio (%) ---
    rounded = np.round(values, 6)
    dry_ratio = np.count_nonzero(rounded == 0) / n * 100 if n > 0 else np.nan

    # --- 4th with most rainfall ---
    if total_acc > 0 and n > 1:
        cum = np.cumsum(values) / total_acc
        time_norm = np.linspace(0, 1, n)

        target_pts = np.linspace(0, 1, 4 + 1)  # 4 bins
        interp_func = interp1d(time_norm, cum, kind='linear', fill_value='extrapolate')

        resampled = interp_func(target_pts)
        fourth_with_most = int(np.argmax(np.diff(resampled)))
    else:
        fourth_with_most = np.nan

    return {
        'total_acc': total_acc,
        'peak_position_ratio': peak_position_ratio,
        'd50': d50,
        'gini': gini,
        'time_based_std': time_based_std,
        'dry_ratio': dry_ratio,
        'max_quartile_index': fourth_with_most
    }






def plot_peak_check(ax, cube, catchment_poly, this_event_results, title):
    """
    Sanity check plot:
    - Full SM field at a given timestep
    - Catchment boundary
    - Peak cell marked as a scatter point
    - Neighbouring cells highlighted to confirm index alignment
    """
    
    x_coord = int(this_event_results['x_coord'])
    y_coord =int(this_event_results['y_coord'])
    x_idx_global = int(this_event_results['x_idx_global'])
    y_idx_global = int(this_event_results['y_idx_global'])
    
    sm_x = cube.coord('projection_x_coordinate').points
    sm_y = cube.coord('projection_y_coordinate').points
    dx   = sm_x[1] - sm_x[0]
    dy   = sm_y[1] - sm_y[0]

    # Full SM field
    iplt.pcolormesh(cube, axes=ax, cmap='Blues')

    # Catchment boundary
    ax.add_geometries(
        [catchment_poly], crs=ccrs.OSGB(),
        facecolor='none', edgecolor='black', linewidth=1.5)

    # Peak cell as a box (so you can see it aligns with the pcolormesh grid)
    peak_box = plt.matplotlib.patches.Rectangle(
        (x_coord - dx/2, y_coord - dy/2), dx, dy,
        linewidth=2, edgecolor='red', facecolor='red', alpha=0.1,
        transform=ccrs.OSGB())
    ax.add_patch(peak_box)

    # Centre point
    ax.scatter(x_coord, y_coord, s=20, color='red', zorder=6,
               transform=ccrs.OSGB(), label=f'Peak ({x_idx_global}, {y_idx_global})')

    # Zoom to catchment with buffer
    minx, miny, maxx, maxy = catchment_poly.bounds
    buf = 20000  # 20km buffer
    ax.set_extent([minx-buf, maxx+buf, miny-buf, maxy+buf], crs=ccrs.OSGB())

#     ax.set_title(f"Peak cell check — global idx ({x_idx_global}, {y_idx_global})\n"
#                  f"coords ({x_coord:.0f}, {y_coord:.0f})")
    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.tight_layout()


def extract_catchment_cube(cube, catchment_poly, method, buffer=0):
    start_time1 = time.time()
    sub_cube, x_offset, y_offset = subset_cube_to_bbox(cube, catchment_poly, buffer)
    end_time1 = time.time()
    time_elapsed1 = end_time1 - start_time1 # round(end_time - start_time,3)
    #print(f"Time for subsetting: {round(time_elapsed1,2)} seconds")
    
    start_time2 = time.time()
    masked_cube = mask_cube_with_catchment(sub_cube, catchment_poly, method)
    end_time2 = time.time()
    time_elapsed2 = end_time2 - start_time2 # round(end_time - start_time,3)
    #print(f"Time for masking: {round(time_elapsed2,2)} seconds")
    
    time_elapsed_total = end_time2 - start_time1 # round(end_time - start_time,3)
    #print(f"Total time:: {round(time_elapsed_total,2)} seconds")
    return masked_cube


