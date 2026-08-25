import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt
import numpy as np
import cartopy.crs as ccrs
import gc
import os
from shapely.ops import unary_union
import iris
import logging
from shapely.geometry import box, MultiPolygon
from scipy.ndimage import label, generate_binary_structure
import cftime

from functions import *
from config import *
    
# def plot_surrounding_ts (rainfall_cube_dir, catchment_num, ens_num, bc=True):
#     catchment_name  = CATCHMENT_LOOKUP_DICT[str(catchment_num)]
#     boundary_gdf    = CATCHMENTS[CATCHMENTS['HA_NUM'] == str(catchment_num)]
#     CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]
#     # catchment_gdf = gpd.GeoDataFrame(geometry=[CATCHMENT_POLY], crs=waterbodies_df.crs)
#     # waterbodies_in_catchment = gpd.sjoin(waterbodies_df, catchment_gdf, how='inner', predicate='within')
#     full_rain_cube = get_rainfall_cube_subsection(2015, '01', f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/01/", 1, 2)
#     FULL_MASK_2D   = mask_cube_with_catchment_full_grid(full_rain_cube[0], CATCHMENT_POLY, method='center_point')
#     year_cube, mask_x_offset, mask_y_offset = subset_cube_to_bbox(full_rain_cube, CATCHMENT_POLY, buffer=0)
#     ny_sub = year_cube.shape[1]
#     nx_sub = year_cube.shape[2]
#     mask_2d_sub = FULL_MASK_2D[mask_y_offset:mask_y_offset + ny_sub, mask_x_offset:mask_x_offset + nx_sub]


#     rainfall_events  = pd.read_csv(RAINFALL_CSV_DIR + f"{catchment_name}_{ens_num}_full_events_with_event_nums.csv")
#     rainfall_events['event_num']  = range(1, len(rainfall_events) + 1)
#     rainfall_events['hydro_year'] = rainfall_events['start_year'].where(rainfall_events['start_month'] != 12,
#             rainfall_events['start_year'] + 1)

#     event_details_cache = {
#         ev: get_rainfall_event_details(rainfall_events, ev)
#         for ev in rainfall_events['event_num']}

#     # ── Get single peak event only ────────────────────────────────────────────
#     peak_event_row = rainfall_events.nlargest(1, 'peaks').iloc[0]
#     year           = int(peak_event_row['hydro_year'])
#     event_num      = int(peak_event_row['event_num'])
#     event_details  = event_details_cache[event_num]

#     # ── Load cube for the peak event, with extra timesteps either side ────────
#     BUFFER = 3
#     if bc ==True:
#         year_cube = get_rainfall_cube_subsection(
#             year, ens_num, rainfall_cube_dir,
#             event_details['start_idx'] - 1 - BUFFER,
#             event_details['stop_idx']  + BUFFER)
#     else:
#         year_cube = get_rainfall_cube_subsection_notbc(
#             year, ens_num, rainfall_cube_dir,
#             event_details['start_idx'] - 1 - BUFFER,
#             event_details['stop_idx']  + BUFFER)   
    
#     year_cube.data = np.where(FULL_MASK_2D, year_cube.data, np.nan)
#     year_cube, x_offset, y_offset = subset_cube_to_bbox(year_cube, CATCHMENT_POLY, buffer=0)

#     # ── Find the peak timestep index within this cube ─────────────────────────
#     max_time_idx = np.unravel_index(np.nanargmax(year_cube.data), year_cube.data.shape)[0]

#     # ── Build the 7 timestep indices to plot (clamped to cube bounds) ─────────
#     n_times      = year_cube.shape[0]
#     offsets      = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4,5,6 ]
#     plot_indices = [np.clip(max_time_idx + o, 0, n_times - 1) for o in offsets]

#     # ── Validate event ────────────────────────────────────────────────────────
#     this_event_results = find_max_precip_location_new(
#         year_cube, event_details['start_idx'], event_details['stop_idx'],
#         x_offset=x_offset, y_offset=y_offset, mask_2d=mask_2d_sub)
#     this_event_results['max_precip_from_csv'] = event_details['max_precip_from_csv']

#     time_coord    = year_cube.coord('time')
#     time_from_cube = time_coord.units.num2date(np.floor(time_coord.points[0]))
#     time_from_csv  = cftime.Datetime360Day(
#         event_details['yr'], event_details['month'],
#         event_details['day'], event_details['hour'])
#     this_event_results['time_mismatch'] = (time_from_cube != time_from_csv)


#     #Get 1D cube, just at the location of the peak
#     rainfall_at_peak  = get_data_at_peak_cell(year_cube, this_event_results, 'x_idx', 'y_idx')
    
#     #Extract various temporal profile results
#     temp_profile_dict = find_temporal_profile_new(rainfall_at_peak, this_event_results, plot=True)

#     # ── Plot: 7 subplots centred on the peak timestep ─────────────────────────
#     fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(18, 9),
#                             subplot_kw={'projection': ccrs.OSGB()})
#     axs = axs.flatten()

#     for ax_i, (offset, t_idx) in enumerate(zip(offsets, plot_indices)):
#         label = "peak" if offset == 0 else f"t{offset:+d}"

#         # Max precip value across the catchment at this timestep
#         timestep_max = np.nanmax(year_cube[t_idx, :, :].data)
#         iplt.pcolormesh(year_cube[t_idx, :, :], axes=axs[ax_i],
#                         cmap='YlGn', edgecolor='none', linewidth=0.5)
#         axs[ax_i].coastlines(resolution='10m', color='red')
#         axs[ax_i].add_geometries([CATCHMENT_POLY], crs=ccrs.OSGB(),
#                                   facecolor='none', edgecolor='black', linewidth=1.5)
#         axs[ax_i].set_title(f"{label} — max: {timestep_max:.1f} mm")

#     # Hide the unused 8th subplot (4×2 grid has 8 slots, we use 7)
#     # axs[7].set_visible(False)

#     fig.suptitle(f"Catchment {catchment_num} — EM{ens_num} — Peak event (year {year}, event {event_num})",
#                  fontsize=12)
#     fig.tight_layout()
    


def get_exceedance_summary(year_cube, peak_t_idx, threshold=150, window=10):
    """
    For the ±window timesteps around peak_t_idx, count how many
    (timestep, grid_cell) combinations exceed the threshold.
    
    Returns a dict with:
      - exceedance_matrix : 2D array (timesteps x cells) of booleans
      - cells_per_timestep: how many cells exceed threshold at each timestep
      - timesteps_per_cell: how many timesteps each cell exceeds threshold
      - n_timesteps_any_exceedance: how many timesteps have at least 1 cell over threshold
      - n_cells_any_exceedance   : how many cells exceed threshold in at least 1 timestep
      - total_exceedances        : total (t, cell) pairs exceeding threshold
    """
    n_times = year_cube.shape[0]

    # Build window indices, clamped to cube bounds
    t_start = max(0, peak_t_idx - window)
    t_end   = min(n_times - 1, peak_t_idx + window)
    window_indices = list(range(t_start, t_end + 1))

    # Extract windowed data: shape (n_window_timesteps, ny, nx)
    window_data = year_cube[window_indices, :, :].data  # shape: (T, Y, X)

    # Flatten spatial dims → shape (T, n_cells)
    T, ny, nx  = window_data.shape
    flat_data  = window_data.reshape(T, -1)             # (T, n_cells)

    # Boolean exceedance matrix
    exceedance_matrix = flat_data > threshold           # (T, n_cells)

    # Per-timestep: how many cells exceed threshold
    cells_per_timestep = exceedance_matrix.sum(axis=1)  # length T

    # Per-cell: how many timesteps exceed threshold
    timesteps_per_cell = exceedance_matrix.sum(axis=0)  # length n_cells
    # Reshape back to spatial grid for mapping
    timesteps_per_cell_2d = timesteps_per_cell.reshape(ny, nx)

    summary = {
        "threshold":                  threshold,
        "window":                     window,
        "t_window_start":             t_start,
        "t_window_end":               t_end,
        "peak_t_idx":                 peak_t_idx,
        "cells_per_timestep":         cells_per_timestep,       # array, length T
        "timesteps_per_cell_2d":      timesteps_per_cell_2d,    # 2D grid
        "n_timesteps_any_exceedance": int((cells_per_timestep > 0).sum()),
        "n_cells_any_exceedance":     int((timesteps_per_cell > 0).sum()),
        "total_exceedances":          int(exceedance_matrix.sum()),
        "max_cells_in_one_timestep":  int(cells_per_timestep.max()),
        "max_timesteps_one_cell":     int(timesteps_per_cell.max()),
    }
    return summary, window_indices    


def find_max_precip_location_new(cube, start_idx, stop_idx, x_offset=0, y_offset=0, mask_2d=None):
    
    # Slice the event window first — much smaller than full year
    #cube_sliced = cube[start_idx:stop_idx,:,:]
    data = np.array(cube.data)
    data[data >= 1e19] = np.nan

    # Then apply mask only to this small slice
    if mask_2d is not None:
        data = np.where(mask_2d, data, np.nan)

    flat_index = np.nanargmax(data)
    # print(f"Max precip: {np.nanmax(data)}")
    t_local, y_idx, x_idx = np.unravel_index(flat_index, data.shape)

    return {
        'max_precip':   float(data[t_local, y_idx, x_idx]),
        't_global':     int(t_local + start_idx),
        't_local':      int(t_local),
        'x_idx':        int(x_idx),
        'y_idx':        int(y_idx),
        'x_idx_global': int(x_idx + x_offset),
        'y_idx_global': int(y_idx + y_offset),
        'x_coord':      cube.coord('projection_x_coordinate').points[x_idx],
        'y_coord':      cube.coord('projection_y_coordinate').points[y_idx],}


def find_temporal_profile_new(cube, details, plot):
    values = cube.data
    
    # Stay in cftime — don't use yyyymmddhh at all
    time_coord = cube.coord('time')
    times_cftime = [time_coord.units.num2date(t) for t in time_coord.points]
    
    metrics = compute_temporal_metrics(values)
    temp_profile_dict = {**metrics, 'times': times_cftime, 'values': values}
    
    if plot:
        # Use integer indices for x-axis to avoid Gregorian calendar issues
        x = np.arange(len(values))
        tick_labels = [f"{t.month:02d}-{t.day:02d} {t.hour:02d}:00" for t in times_cftime]
        
        plt.figure(figsize=(8, 4))
        plt.plot(x, values, color='black')
        
        # Mark peak
        plt.axvline(details['t_local'], linestyle='--', color='red', label='Peak timestep')
        
        # Format ticks — show every N ticks to avoid crowding
        N = max(1, len(x) // 8)
        plt.xticks(x[::N], tick_labels[::N], rotation=45, ha='right')
        
        plt.xlabel("Date")
        plt.ylabel("Precipitation intensity (mm/hr)")
        plt.title("Rainfall at peak grid cell during event")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return temp_profile_dict

def setup_worker_logger(catchment_num):
    """Each worker logs to its own file so output never interleaves."""
    log_path = os.path.join(OUT_DIR, f"logs/catchment_{catchment_num}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = logging.getLogger(f"catchment_{catchment_num}")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers (important if function is called multiple times)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    
    return logger


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
            'global_end':   cumulative + HOURS_PER_MONTH
        })
        cumulative += HOURS_PER_MONTH

    for m in range(1, 12):
        all_files.append({
            'path':         f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr}{m:02d}01-{yr}{m:02d}30.nc",
            'global_start': cumulative,
            'global_end':   cumulative + HOURS_PER_MONTH
        })
        cumulative += HOURS_PER_MONTH

    all_files.append({
        'path':         f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr}1201-{yr}1230.nc",
        'global_start': cumulative,
        'global_end':   cumulative + HOURS_PER_MONTH
    })

    if start_idx is not None and stop_idx is not None:
        if not has_prev_december and start_idx < HOURS_PER_MONTH:
            raise ValueError(
                f"Event requests start_idx={start_idx} (before 1 Jan {yr}), but no "
                f"December {yr-1} file exists — dataset begins {DATASET_MIN_YEAR}1201. "
                f"This event cannot be loaded."
            )
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


# def get_rainfall_cube_subsection(yr, ENS_NUM, RAINFALLDIR, start_idx=None, stop_idx=None):
#     """
#     Load a subsection of hourly rainfall data for a given hydrological year and ensemble member.
    
#     The hydrological year is defined as December(yr-1) through November(yr), meaning:
#         - Index 0    = 1 Dec (yr-1) 00:00
#         - Index 719  = 30 Dec (yr-1) 23:00  [end of December]
#         - Index 720  = 1 Jan (yr)   00:00
#         - Index 8639 = 30 Nov (yr)  23:00   [end of November, end of hydro year]
    
#     Each monthly file contains exactly 720 timesteps (30-day months x 24 hours).
#     This is a 360-day calendar dataset so every month has exactly 30 days.
    
#     If start_idx and stop_idx are provided, only the files needed to cover that
#     window are loaded — typically 1 or 2 files instead of all 12. This is important
#     for performance when looping over many events.
    
#     An additional December(yr) file is included at the end of the file list to handle
#     edge cases where an event starting in late November bleeds past the end of the
#     hydrological year (index > 8639).
    
#     Parameters
#     ----------
#     yr         : int  — the hydrological year (e.g. 2067 means Dec 2066 → Nov 2067)
#     ENS_NUM    : str  — ensemble member identifier (e.g. '01')
#     RAINFALLDIR: str  — path to directory containing monthly .nc files
#     start_idx  : int  — first timestep to load (inclusive), in hydro-year index space
#     stop_idx   : int  — last timestep to load (exclusive), in hydro-year index space
    
#     Returns
#     -------
#     iris.cube.Cube with dimensions (time, projection_y_coordinate, projection_x_coordinate)
#     """
#     HOURS_PER_MONTH = 30 * 24  # 720 — constant for this 360-day calendar dataset

#     # ── Build the full ordered list of monthly files ───────────────────────────
#     # Each entry records the file path and the global index range it covers,
#     # so we can later identify which files overlap with [start_idx, stop_idx].
#     # 'global_start' is inclusive, 'global_end' is exclusive (half-open interval).
#     all_files  = []
#     cumulative = 0  # running total of timesteps, used to assign global index ranges

#     # File 0: December of the previous calendar year
#     # This is always the first month of the hydrological year (indices 0–719)
#     all_files.append({
#         'path':         f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr-1}1201-{yr-1}1230.nc",
#         'global_start': cumulative,
#         'global_end':   cumulative + HOURS_PER_MONTH
#     })
#     cumulative += HOURS_PER_MONTH

#     # Files 1–11: January through November of the target year (indices 720–8639)
#     # Note: we stop at month 11 (November) because December belongs to the
#     # *next* hydrological year
#     for m in range(1, 12):
#         all_files.append({
#             'path':         f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr}{m:02d}01-{yr}{m:02d}30.nc",
#             'global_start': cumulative,
#             'global_end':   cumulative + HOURS_PER_MONTH
#         })
#         cumulative += HOURS_PER_MONTH

#     # File 12: December of the target year (indices 8640–9359)
#     # This file is ONLY needed for edge-case events that start in late November
#     # and whose stop_idx bleeds past the end of the hydrological year (> 8639).
#     # Including it here costs nothing if it isn't needed — the filter below will
#     # simply not select it.
#     all_files.append({
#         'path':         f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr}1201-{yr}1230.nc",
#         'global_start': cumulative,
#         'global_end':   cumulative + HOURS_PER_MONTH
#     })

#     # ── Select only the files that overlap with [start_idx, stop_idx] ─────────
#     # A file overlaps the requested window if:
#     #   - it ends after the window starts (global_end > start_idx), AND
#     #   - it starts before the window ends (global_start < stop_idx)
#     # This is standard half-open interval intersection logic.
#     # For a typical event this reduces 13 file loads down to 1 or 2.
#     if start_idx is not None and stop_idx is not None:
#         needed = [f for f in all_files
#                   if f['global_end'] > start_idx and f['global_start'] < stop_idx]
#     else:
#         # No indices provided — load the full hydrological year
#         needed = all_files

#     # ── Warn if the overflow December file is being used ──────────────────────
#     # This is an unusual case and worth flagging so we know it's happening
#     #if needed and needed[-1]['path'].endswith(f"{yr}1201-{yr}1230.nc"):
#     #    print(f"  WARNING: event bleeds into Dec {yr} (stop_idx={stop_idx} > 8639) — loading overflow file")

#     # ── Load only the needed files and concatenate into a single cube ─────────
#     monthly_cubes = iris.cube.CubeList()
#     for f in needed:
#         cube = iris.load(f['path'])[1]  # [1] selects the rainfall variable from the file
#         cube.attributes = {}            # clear attributes so concatenation doesn't fail
#                                         # on mismatched metadata between monthly files
#         monthly_cubes.append(cube)

#     for cube in monthly_cubes:
#         cube.coord('time').bounds = None
        
#     year_cube = monthly_cubes.concatenate_cube()

#     # ── Slice the concatenated cube to the exact requested window ─────────────
#     # The indices stored in start_idx/stop_idx are in the global hydro-year
#     # index space (0 = start of Dec(yr-1)). But after loading only a subset
#     # of files, the cube's own time axis starts at the beginning of the first
#     # *loaded* file, not at 0. So we subtract that file's global_start ('offset')
#     # to convert global indices into local indices within the loaded cube.
#     if start_idx is not None and stop_idx is not None:
#         offset = needed[0]['global_start']
#         return year_cube[start_idx - offset:stop_idx - offset, :, :]

#     return year_cube

def get_data_at_peak_cell(cube, one_event_results, x_idx_variable, y_idx_variable):

    x_idx = one_event_results[x_idx_variable]
    y_idx = one_event_results[y_idx_variable]

    x_coord = one_event_results["x_coord"]
    y_coord = one_event_results["y_coord"]

    cube_x = cube.coord('projection_x_coordinate').points
    cube_y = cube.coord('projection_y_coordinate').points

    # Check the coordinate at the global index matches what we stored
    assert np.isclose(cube_x[x_idx], x_coord, atol=100), \
        f"x mismatch: {cube_x[x_idx]} vs {x_coord}"
    assert np.isclose(cube_y[y_idx], y_coord, atol=100), \
        f"y mismatch: {cube_y[y_idx]} vs {y_coord}"

    # Extract timeseries
    if len(cube.shape) == 2:
        cube_at_peak = cube[y_idx, x_idx]
    elif len(cube.shape) == 3:
        cube_at_peak = cube[:, y_idx, x_idx]

    return cube_at_peak    

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


def run_spatial_diagnostics(row, event_num, sm, flood_cubes, catchment_poly,
                             ens_num, rainfall_cube_dir, FULL_MASK_2D, depth=10):
    """
    Plots a 2x2 spatial check for a single event.
    Rainfall cube is loaded here on demand and discarded after plotting.

    Parameters
    ----------
    row               : namedtuple   — single event row from itertuples()
    event_num         : int          — event number (passed explicitly)
    sm                : iris.Cube    — soil moisture cube for this (ens, year)
    HC_CUBE           : iris.Cube    — hydraulic conductivity (static)
    flood_cubes       : dict         — pre-loaded flood cubes, keyed [ens_num][depth]
    catchment_poly    : shapely geom — catchment boundary polygon
    ens_num           : str          — ensemble member ID
    rainfall_cube_dir : str          — directory for hourly rainfall cubes
    depth             : int          — flood depth threshold to plot (10 or 30)
    """
    # Load and slice rainfall cube only when needed
    rainfall_yr_cube = get_rainfall_cube(row['start_year'], row['ens'], rainfall_cube_dir)
    rainfall_slice   = rainfall_yr_cube[row['start_idx']:row['stop_idx'], :, :][row['t_local'], :, :]
    rainfall_slice.data = np.where(FULL_MASK_2D, rainfall_slice.data, np.nan)
    
    sm_slice    = sm[row['hydro_day_360'], :, :]
    flood_slice = flood_cubes[depth]['area']
    flood_slice = flood_slice[event_num-1, :,:]
    
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 8),
                            subplot_kw={'projection': ccrs.OSGB()})
    axs = axs.flatten()

    plot_peak_check(axs[0], sm_slice,    catchment_poly, row, title='Soil moisture')
    plot_peak_check(axs[1], rainfall_slice, catchment_poly, row, title='Precipitation')
    # plot_peak_check(axs[2], HC_CUBE,     catchment_poly, row, title='Hydraulic conductivity')
    plot_peak_check(axs[3], flood_slice, catchment_poly, row, title=f'Flooded area ({depth}cm)')

    fig.suptitle(
        f"Spatial diagnostics — ens={ens_num}, year={row['start_year']}, event={event_num}\n"
        f"Peak day: {row['rainfall_peak_day']}",
        fontsize=12)
    plt.tight_layout()
    plt.show()

    # Discard — no reason to keep it in memory after plotting
    del rainfall_yr_cube
    gc.collect()   

def maybe_diagnose(row, event_num, condition=False, **kwargs):
    """
    Calls run_spatial_diagnostics only if condition is True.
    event_num is passed explicitly rather than read from row.
    """
    if condition:
        run_spatial_diagnostics(row, event_num=event_num, **kwargs)
        
        
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
    fig, axes = plt.subplots(
        1, ncols, figsize=(5 * ncols, 5),
        subplot_kw={'projection': ccrs.OSGB()}
    )
    axes = list(axes)

    panels = [
        ("Threshold mask\n"
         f"(>= {threshold_level*100:.0f}% of peak)",  threshold_polys, axes[0]),
        ("Peak cluster\n"
         f"neighbourhood sum = {neighbourhood_sum:.2f} mm", cluster_polys, axes[1]),
    ]

    # ── Rainfall panels ───────────────────────────────────────────────────────
    for title, polys, ax in panels:
        # Use qplt so iris handles the coordinate-aware pcolormesh
        qplt.pcolormesh(peak_slice, axes=ax, cmap='Blues')

        # Draw the mask outline polygons (threshold or cluster)
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color='orange', linewidth=2, transform=ccrs.OSGB())

        # Mark the peak cell
        ax.scatter(row['x_coord'], row['y_coord'], s=50, color='red', zorder=6,
                   transform=ccrs.OSGB(),
                   label=f"Peak ({int(row['x_idx_global'])},{int(row['y_idx_global'])})")

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
        ax.scatter(row['x_coord'], row['y_coord'], s=50, color='red',
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
        f"Peak = {row['max_precip']:.2f} mm, threshold = {threshold_level*100:.0f}%"
        f" ({row['max_precip'] * threshold_level:.2f} mm)",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()


# def analyse_peak_event(
#     this_event_results,
#     year_cube,
#     flood_numpy,
#     flood_cubes,          # only needed for plotting
#     CATCHMENT_POLY,       # only needed for plotting
#     boundary_gdf,         # only needed for plotting
#     neighbourhood_size,
#     threshold_levels,
#     plot=False
# ):
#     """
#     Fast version: uses preloaded cubes/arrays (no disk access).
#     """

#     peak_value = this_event_results['max_precip']
#     t_local    = int(this_event_results['t_local'])
#     yi         = int(this_event_results['y_idx'])
#     xi         = int(this_event_results['x_idx'])

#     # ── Extract rainfall slice (NO LOADING) ───────────────────────────────
#     peak_slice = year_cube[t_local, :, :]
#     rain = peak_slice.data.astype(float)

#     # Clean invalid values
#     rain = np.where((rain == -99999) | (rain > 1e19), np.nan, rain)

#     # ── Pre-extract flood arrays (already numpy) ──────────────────────────
#     flood_arrays = {
#         (depth, var): flood_numpy[depth][var][this_event_results['event_num'] - 1, :, :]
#         for depth in [10, 30]
#         for var in ['area', 'vol']
#     }

#     # ── Neighbourhood stats ───────────────────────────────────────────────
#     n  = neighbourhood_size
#     y0 = max(0, yi - n);  y1 = min(rain.shape[0], yi + n + 1)
#     x0 = max(0, xi - n);  x1 = min(rain.shape[1], xi + n + 1)

#     neighbourhood = rain[y0:y1, x0:x1]

#     results = {
#         'neighbourhood_rain_total': float(np.nansum(neighbourhood)),
#         'neighbourhood_rain_sum_excl_peak': float(np.nansum(neighbourhood) - peak_value),
#         'neighbourhood_n_cells': int(np.sum(~np.isnan(neighbourhood))),
#     }

#     # Flood totals in neighbourhood
#     for depth in [10, 30]:
#         for var in ['area', 'vol']:
#             results[f'neighbourhood_flood_{depth}_{var}'] = float(
#                 np.nansum(flood_arrays[(depth, var)][y0:y1, x0:x1])
#             )

#     # ── Threshold + cluster stats ─────────────────────────────────────────
#     for threshold_level in threshold_levels:

#         t_suffix = f"_t{int(threshold_level * 100)}"
#         threshold_value = peak_value * threshold_level

#         threshold_mask = (rain >= threshold_value) & (~np.isnan(rain))

#         labeled, _ = label(threshold_mask)
#         peak_label = labeled[yi, xi]
#         cluster_mask = (labeled == peak_label)

#         # Rain stats
#         results[f'threshold_rain_total{t_suffix}'] = float(np.nansum(rain[threshold_mask]))
#         results[f'threshold_n_cells{t_suffix}']    = int(np.sum(threshold_mask))
#         results[f'cluster_rain_total{t_suffix}']   = float(np.nansum(rain[cluster_mask]))
#         results[f'cluster_n_cells{t_suffix}']      = int(np.sum(cluster_mask))

#         # Flood stats
#         for depth in [10, 30]:
#             for var in ['area', 'vol']:
#                 fa = flood_arrays[(depth, var)]
#                 results[f'threshold_flood_{depth}_{var}{t_suffix}'] = float(np.nansum(fa[threshold_mask]))
#                 results[f'cluster_flood_{depth}_{var}{t_suffix}']   = float(np.nansum(fa[cluster_mask]))
        
#         # ── Optional plotting ─────────────────────────────────────────────
#         if plot and threshold_level == 0.5:
#             plot_cluster_check(
#                 rain=rain,
#                 peak_slice=peak_slice,
#                 threshold_mask=threshold_mask,
#                 cluster_mask=cluster_mask,
#                 row=this_event_results,
#                 y0=y0, y1=y1, x0=x0, x1=x1,
#                 threshold_level=threshold_level,
#                 neighbourhood_sum=results['neighbourhood_rain_sum_excl_peak'],
#                 neighbourhood_size=neighbourhood_size,
#                 flood=flood_cubes[10]['area'][this_event_results['event_num'] - 1],
#                 catchment_poly=CATCHMENT_POLY,
#                 boundary_gdf=boundary_gdf,
#             )
            
            
#     assert cluster_mask[yi, xi] == True            
#     assert results['neighbourhood_n_cells'] > 0
#     assert results['neighbourhood_n_cells'] <= (2*n+1)**2
#     assert results['cluster_rain_total_t50'] <= results['threshold_rain_total_t50']
    
#     return results

def analyse_peak_event(
    this_event_results,
    year_cube,
    flood_numpy,
    flood_cubes,          # only needed for plotting (iris cubes retain coords for maps)
    CATCHMENT_POLY,       # only needed for plotting
    boundary_gdf,         # only needed for plotting
    neighbourhood_size,
    threshold_levels,
    antecedent_arrays=None,   # dict: {'ante_rain_1d': 2D array, 'ante_rain_5d': 2D array, 'ante_sm': 2D array, ...}
    plot=False
):
    """
    Characterise a single rainfall event's spatial footprint at three
    nested spatial scales, and summarise flood outcomes and antecedent
    wetness conditions (rainfall and soil moisture) over each of those
    same footprints.

    The three spatial scales, from smallest/fixed to largest/event-defined:

    1. Peak cell (yi, xi)
       — the single grid cell where the event's maximum precipitation
         occurred. Used as the indexing anchor for the other two scales,
         and directly for the '..._point' antecedent variables.

    2. Neighbourhood
       — a fixed-size square window of (2*neighbourhood_size + 1) cells
         centred on the peak cell, regardless of how the storm actually
         looked. A crude, storm-agnostic "local area" scale — useful as
         a baseline that doesn't depend on the threshold_levels choice.

    3. Threshold mask / cluster
       — cells where rainfall exceeded `threshold_level * peak_value`.
         `threshold_mask` includes ALL such cells in the catchment (which
         may include unconnected patches elsewhere in the domain that
         happened to also be intense). `cluster_mask` is the subset of
         `threshold_mask` that is spatially CONNECTED to the peak cell
         (via scipy.ndimage.label) — i.e. the actual contiguous storm
         footprint containing the peak, not just "intense cells anywhere".
         This is repeated at each level in `threshold_levels` (e.g. 50%,
         60%, 80% of peak intensity) to see how footprint size and
         antecedent conditions change as you tighten the definition of
         "part of the storm".

    For each of these three scales, the function reports:
      - the rainfall itself (total/n_cells for footprint size and intensity)
      - flood outcomes (area, volume, at 10cm and 30cm depth thresholds)
      - antecedent conditions (from `antecedent_arrays`), summarised as
        the MEAN over the footprint — not the sum, since footprint size
        varies event-to-event and across threshold levels, and we want
        "how wet was this area beforehand" (a rate/state), not "how much
        antecedent rain fell across an area of this size" (which would
        just track footprint size and confound with storm extent).

    Parameters
    ----------
    this_event_results : dict
        Per-event metadata already computed upstream: must contain
        'max_precip', 't_local' (time index of peak within year_cube),
        'y_idx'/'x_idx' (grid location of peak, within year_cube's
        coordinate system), and 'event_num' (1-indexed, used to index
        into the flood arrays' leading (event) dimension).
    year_cube : iris.cube.Cube
        The event-window rainfall cube (time, y, x), already subset to
        the catchment bbox. Only the peak-time slice is used here.
    flood_numpy : dict
        {depth: {'area': ndarray, 'vol': ndarray}} — flood outcome
        arrays, shape (event, y, x), same y/x grid as year_cube.
    antecedent_arrays : dict or None
        Optional. Each value is a 2D (y, x) array on the SAME grid and
        offset as `rain` (i.e. year_cube's spatial subset) — e.g. a
        rolling rainfall total over a lookback window, or the most
        recent soil moisture field before the event. Keys become part
        of the output column names (see below).
    neighbourhood_size : int
        Half-width of the fixed neighbourhood window, in cells.
    threshold_levels : list of float
        e.g. [0.5, 0.6, 0.8] — fractions of peak intensity defining the
        storm footprint at each scale.
    plot : bool
        If True, generates a diagnostic figure at the 50% threshold level
        only (for spot-checking specific events, not for bulk runs).

    Returns
    -------
    dict
        Flat dict of scalar results, suffixed by scale
        (neighbourhood_/threshold_..._t50/cluster_..._t50 etc.),
        ready to be merged into `this_event_results`.
    """

    # ── Pull out the essentials for this event ──────────────────────────────
    peak_value = this_event_results['max_precip']
    t_local    = int(this_event_results['t_local'])
    yi         = int(this_event_results['y_idx'])   # peak cell row
    xi         = int(this_event_results['x_idx'])   # peak cell col

    # ── Extract the rainfall field AT THE MOMENT OF PEAK INTENSITY ──────────
    # (This is a single time-slice, not a total over the event — it's what
    # defines the storm's spatial footprint/shape at its most intense moment.)
    peak_slice = year_cube[t_local, :, :]
    rain = peak_slice.data.astype(float)

    # Clean sentinel/fill values so they don't pollute nansum/nanmean below
    rain = np.where((rain == -99999) | (rain > 1e19), np.nan, rain)

    # ── Pre-slice flood arrays down to this event's 2D (y, x) layer ─────────
    # (flood_numpy is 3D: event x y x x — index into the event dimension once,
    # up front, so the threshold loop below doesn't repeat this lookup.)
    flood_arrays = {
        (depth, var): flood_numpy[depth][var][this_event_results['event_num'] - 1, :, :]
        for depth in [10, 30]
        for var in ['area', 'vol']
    }

    # ══════════════════════════════════════════════════════════════════════
    # SCALE 1: Neighbourhood — fixed-size window around the peak cell
    # ══════════════════════════════════════════════════════════════════════
    n  = neighbourhood_size
    y0 = max(0, yi - n);  y1 = min(rain.shape[0], yi + n + 1)
    x0 = max(0, xi - n);  x1 = min(rain.shape[1], xi + n + 1)
    neighbourhood = rain[y0:y1, x0:x1]

    results = {
        # Total rainfall summed across the neighbourhood window
        'neighbourhood_rain_total': float(np.nansum(neighbourhood)),
        # Same, but excluding the peak cell's own value (isolates the
        # contribution of the surrounding cells only)
        'neighbourhood_rain_sum_excl_peak': float(np.nansum(neighbourhood) - peak_value),
        # How many valid (non-NaN, i.e. in-catchment) cells the window
        # actually covered — window can be clipped near catchment edges
        'neighbourhood_n_cells': int(np.sum(~np.isnan(neighbourhood))),
    }

    # Flood outcomes summed over the same neighbourhood window
    for depth in [10, 30]:
        for var in ['area', 'vol']:
            results[f'neighbourhood_flood_{depth}_{var}'] = float(
                np.nansum(flood_arrays[(depth, var)][y0:y1, x0:x1]))

    # ── Antecedent conditions over the neighbourhood ─────────────────────────
    if antecedent_arrays:
        for ante_name, ante_grid in antecedent_arrays.items():
            ante_nbhd = ante_grid[y0:y1, x0:x1]
            # Mean over the window — "how wet was the local area beforehand"
            results[f'neighbourhood_{ante_name}_mean'] = float(np.nanmean(ante_nbhd))
            # Value at the single peak cell only — the most localised
            # possible antecedent measure, matched 1:1 to where HC/flood
            # stats are also extracted at a point
            results[f'neighbourhood_{ante_name}_point'] = float(ante_grid[yi, xi])

    # ══════════════════════════════════════════════════════════════════════
    # SCALE 2 & 3: Threshold mask and connected cluster, per threshold level
    # ══════════════════════════════════════════════════════════════════════
    for threshold_level in threshold_levels:
        t_suffix = f"_t{int(threshold_level * 100)}"   # e.g. '_t50', '_t60', '_t80'
        threshold_value = peak_value * threshold_level

        # All cells (anywhere in the catchment) whose rainfall meets the
        # threshold — may include disconnected patches unrelated to this storm
        threshold_mask = (rain >= threshold_value) & (~np.isnan(rain))

        # Connected-component labelling: isolates just the patch of
        # threshold-exceeding cells that is spatially contiguous with the
        # peak cell — i.e. the actual storm footprint, not incidental
        # intense cells elsewhere in the catchment
        labeled, _ = label(threshold_mask)
        peak_label = labeled[yi, xi]
        cluster_mask = (labeled == peak_label)

        # Rainfall stats: footprint size (n_cells) and total intensity,
        # for both the loose threshold set and the tight connected cluster
        results[f'threshold_rain_total{t_suffix}'] = float(np.nansum(rain[threshold_mask]))
        results[f'threshold_n_cells{t_suffix}']    = int(np.sum(threshold_mask))
        results[f'cluster_rain_total{t_suffix}']   = float(np.nansum(rain[cluster_mask]))
        results[f'cluster_n_cells{t_suffix}']      = int(np.sum(cluster_mask))

        # Flood outcomes summed over each footprint definition
        for depth in [10, 30]:
            for var in ['area', 'vol']:
                fa = flood_arrays[(depth, var)]
                results[f'threshold_flood_{depth}_{var}{t_suffix}'] = float(np.nansum(fa[threshold_mask]))
                results[f'cluster_flood_{depth}_{var}{t_suffix}']   = float(np.nansum(fa[cluster_mask]))

        # ── Antecedent conditions over each footprint definition ────────────
        if antecedent_arrays:
            for ante_name, ante_grid in antecedent_arrays.items():
                # Mean antecedent value across ALL threshold-exceeding cells
                # (includes any disconnected intense patches elsewhere)
                results[f'threshold_{ante_name}_mean{t_suffix}'] = float(np.nanmean(ante_grid[threshold_mask]))
                # Mean antecedent value across just the connected storm
                # footprint — the most physically meaningful of the three
                # antecedent scales, since it's matched exactly to "the
                # area that actually received this storm's core rainfall"
                results[f'cluster_{ante_name}_mean{t_suffix}']   = float(np.nanmean(ante_grid[cluster_mask]))

        # ── Optional diagnostic plot, generated once at the 50% level only ──
        if plot and threshold_level == 0.5:
            plot_cluster_check(
                rain=rain, peak_slice=peak_slice,
                threshold_mask=threshold_mask, cluster_mask=cluster_mask,
                row=this_event_results, y0=y0, y1=y1, x0=x0, x1=x1,
                threshold_level=threshold_level,
                neighbourhood_sum=results['neighbourhood_rain_sum_excl_peak'],
                neighbourhood_size=neighbourhood_size,
                flood=flood_cubes[10]['area'][this_event_results['event_num'] - 1],
                catchment_poly=CATCHMENT_POLY, boundary_gdf=boundary_gdf)

    # ── Sanity checks ─────────────────────────────────────────────────────
    # (These use the LAST threshold_level's cluster_mask from the loop above,
    # since Python leaves loop variables bound after the loop exits — a bit
    # implicit; worth being aware this only checks the final threshold level,
    # not all of them.)
    assert cluster_mask[yi, xi] == True             # peak cell must always be part of its own cluster
    assert results['neighbourhood_n_cells'] > 0      # window wasn't entirely out-of-catchment
    assert results['neighbourhood_n_cells'] <= (2*n+1)**2  # window wasn't double-counted/malformed
    assert results['cluster_rain_total_t50'] <= results['threshold_rain_total_t50']  # cluster ⊆ threshold set

    return results

       