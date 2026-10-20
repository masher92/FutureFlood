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
    

HOURS_PER_MONTH = 30 * 24  # 720, matches get_rainfall_cube_subsection    
def get_rainfall_lookback_cube(year, ens_num, rainfall_cube_dir, event_start_idx, lookback_hours):
    """
    Fetch a rainfall cube covering [event_start_idx - lookback_hours, event_start_idx)
    in the same global-index convention as get_rainfall_cube_subsection(year, ...).

    Handles the case where the lookback window crosses back before the start
    of `year`'s cube (i.e. before 1 Dec of year-1) by pulling the tail of
    year-1's own cube. Assumes lookback_hours < HOURS_PER_MONTH (true for
    windows up to ~29 days), so at most one boundary crossing is needed.
    """
    if lookback_hours >= HOURS_PER_MONTH:
        raise ValueError(
            f"lookback_hours={lookback_hours} >= {HOURS_PER_MONTH}; this helper "
            f"only handles a single hydro-year boundary crossing.")

    lookback_start_idx = event_start_idx - lookback_hours

    if lookback_start_idx >= 0:
        # Entirely within this year's cube — no boundary crossing needed
        return get_rainfall_cube_subsection(
            year, ens_num, rainfall_cube_dir,
            start_idx=lookback_start_idx, stop_idx=event_start_idx)

    # Need to reach back into year-1's own cube
    if year - 1 <= DATASET_MIN_YEAR:
        raise ValueError(
            f"Lookback window for year={year} needs data before "
            f"{DATASET_MIN_YEAR}1201, which is before the dataset start."
        )

    deficit = -lookback_start_idx  # hours needed from before this year's cube starts

    # Work out year-1's own cube length so we know where its Dec(year-1)
    # block (the tail) sits in its own index space
    prev_has_prev_december = not (year - 1 == DATASET_MIN_YEAR)
    prev_total_len = (13 if prev_has_prev_december else 12) * HOURS_PER_MONTH

    prev_cube = get_rainfall_cube_subsection(
        year - 1, ens_num, rainfall_cube_dir,
        start_idx=prev_total_len - deficit, stop_idx=prev_total_len)

    this_cube = get_rainfall_cube_subsection(
        year, ens_num, rainfall_cube_dir,
        start_idx=0, stop_idx=event_start_idx)

    for c in (prev_cube, this_cube):
        c.attributes = {}
        c.coord('time').bounds = None

    return iris.cube.CubeList([prev_cube, this_cube]).concatenate_cube()

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
                             ens_num, rainfall_cube_dir, FULL_MASK_2D, year_cube, depth=10):
    
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
    rainfall_slice   = year_cube[row['t_local'], :, :]
    # rainfall_slice.data = np.where(FULL_MASK_2D, rainfall_slice.data, np.nan)
    
    sm_slice    = sm[row['hydro_day_360'], :, :]
    flood_slice = flood_cubes[depth]['area']
    flood_slice = flood_slice[event_num-1, :,:]
    
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 8),
                            subplot_kw={'projection': ccrs.OSGB()})
    axs = axs.flatten()

    plot_peak_check(axs[0], sm_slice,    catchment_poly, row, title='Soil moisture')
    plot_peak_check(axs[1], rainfall_slice, catchment_poly, row, title='Precipitation')
    # plot_peak_check(axs[2], HC_CUBE,     catchment_poly, row, title='Hydraulic conductivity')
    plot_peak_check(axs[2], flood_slice, catchment_poly, row, title=f'Flooded area ({depth}cm)')

    fig.suptitle(
        f"Spatial diagnostics — ens={ens_num}, year={row['start_year']}, event={event_num}\n"
        f"Peak day: {row['rainfall_peak_day']}",
        fontsize=12)
    plt.tight_layout()
    plt.show()

    # Discard — no reason to keep it in memory after plotting
    # del rainfall_yr_cube
    gc.collect()    


def maybe_diagnose(row, event_num, condition=False, **kwargs):
    """
    Calls run_spatial_diagnostics only if condition is True.
    event_num is passed explicitly rather than read from row.
    """
    if condition:
        run_spatial_diagnostics(row, event_num=event_num, **kwargs)
        
        


def analyse_peak_event(
    this_event_results,
    year_cube,
    flood_numpy,
    flood_cubes,          # only needed for plotting (iris cubes retain coords for maps)
    CATCHMENT_POLY,       # only needed for plotting
    boundary_gdf,         # only needed for plotting
    neighbourhood_size,
    threshold_levels,
    plot=False
):


    # ── Pull out the essentials for this event ──────────────────────────────
    peak_value = this_event_results['rainpeak_peak_value']
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

       