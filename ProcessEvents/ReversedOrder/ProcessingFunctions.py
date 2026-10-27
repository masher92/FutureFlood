import pandas as pd
import numpy as np
import iris
import glob
import sys
import cftime
from scipy.ndimage import label, generate_binary_structure
from shapely.geometry import box, MultiPolygon
import cartopy.crs as ccrs

sys.path.insert(1, '../')
from HelperFunctions import *

def load_flood_arrays(FLOOD_ROOT, catchment_num, ens_num, mask, catchment_poly,
                      depths=[10, 30], n_events=None):
    """
    Load flooded area and volume for one catchment and ensemble member.

    Returns
    -------
    flood_cubes : {depth: {'area': cube, 'vol': cube}}
    flood_numpy : {depth: {'area': ndarray, 'vol': ndarray}}
        Arrays are (event, y, x). Axis 0 is event_num - 1.
    """
    ens_dir = f"{FLOOD_ROOT}Catchment_{catchment_num}/Ens{ens_num}_{catchment_num}/"

    flood_cubes, flood_numpy = {}, {}
    for depth in depths:
        flood_cubes[depth] = {}
        flood_numpy[depth] = {}
        # dictionary key -> variable name used in the filename
        for key, name in [('area', 'area'), ('vol', 'volume')]:
            fp = (f"{ens_dir}/{depth}cm/flooded_{name}_5km_total_"
                  f"Ens{ens_num}_{catchment_num}_{depth}cm.nc")
            cube = prepare_flood_cube(load_3d_cube(fp))
            cube.data = np.where(mask, cube.data, np.nan)
            cube = filter_closer_to_catchment(cube, catchment_poly, plot=False)

            flood_cubes[depth][key] = cube
            flood_numpy[depth][key] = cube.data

    # sanity checks
    shapes = {arr.shape for d in flood_numpy.values() for arr in d.values()}
    assert len(shapes) == 1, f"Flood arrays have inconsistent shapes: {shapes}"
    if n_events is not None:
        n_flood = next(iter(shapes))[0]
        assert n_flood == n_events, \
            f"Flood file has {n_flood} events but the events CSV has {n_events}"

    return flood_cubes, flood_numpy


def load_rainfall_events(RAINFALL_CSV_DIR, catchment_name, ens_num):
    """"""
    rainfall_events  = pd.read_csv(RAINFALL_CSV_DIR + f"{catchment_name}_{ens_num}_full_events_with_event_nums.csv")
    return rainfall_events  

def add_hydro_year(events):
    """Number events 1..N and assign hydro year (Dec events belong to the next year)."""
    events['event_num']  = range(1, len(events) + 1)
    events['hydro_year'] = events['start_year'] + (events['start_month'] == 12).astype(int)
    return events

def find_wet_onset(values, time_coord, wet_thresh):
    """First/last wet hour (index into the cube) and onset date, or None if never wet."""
    wet = np.flatnonzero(values > wet_thresh)
    if wet.size == 0:
        return None
    first, last = int(wet[0]), int(wet[-1])
    t = time_coord.units.num2date(time_coord.points[first])
    return first, last, (t.year, t.month, t.day)

def sm_day_before(sm_data, day_index, onset_day):
    """(y, x) SM field for the day before onset_day, or NaNs if unavailable."""
    idx = day_index.get(onset_day)
    if idx is None or idx == 0:
        return np.full(sm_data.shape[1:], np.nan)
    return sm_data[idx - 1]

def build_catchment_mask(full_rain_cube, CATCHMENT_POLY):
    """"""
    
    FULL_MASK_2D   = mask_cube_with_catchment_full_grid(full_rain_cube[0], CATCHMENT_POLY, method='center_point')
    year_cube, mask_x_offset, mask_y_offset = subset_cube_to_bbox(full_rain_cube, CATCHMENT_POLY, buffer=0)
    ny_sub = year_cube.shape[1]
    nx_sub = year_cube.shape[2]
    mask_2d_sub = FULL_MASK_2D[mask_y_offset:mask_y_offset + ny_sub, 
                                mask_x_offset:mask_x_offset + nx_sub]
    del full_rain_cube, year_cube
    return FULL_MASK_2D, mask_2d_sub

from collections import namedtuple
SoilMoisture = namedtuple('SoilMoisture', ['cube', 'data', 'times', 'day_index'])
def load_soil_moisture(SM_ROOT, events_in_year, ens_num, hydro_year, catchment_poly, mask):
    """
    Load daily soil moisture for one hydro year (1 Dec to 30 Nov), masked and cropped
    to the catchment. If any event starts on 1 December, the previous year's file is
    prepended so the "day before" lookup works.

    Returns a SoilMoisture with: cube, data (t, y, x), times, day_index {(y, m, d): i}.
    """
    sm_dir  = SM_ROOT + f'Ens_{ens_num}/'
    sm_file = glob.glob(f"{sm_dir}/r001i1p*****_{hydro_year-1}1201-{hydro_year}1130_mrso.nc")[0]
    sm      = iris.load(sm_file)[0]

    needs_extra_sm = ((events_in_year['start_month'] == 12) & (events_in_year['start_day'] == 1)).any()
    if needs_extra_sm:
        prev_files = glob.glob(f"{sm_dir}/r001i1p*****_{hydro_year-2}1201-{hydro_year-1}1130_mrso.nc")
        if prev_files:
            prev_sm = iris.load(prev_files[0])[0]
            prev_sm.attributes = {}
            sm.attributes = {}
            sm = iris.cube.CubeList([prev_sm, sm]).concatenate_cube()
        else:
            print(f"  WARNING: prior SM file for {hydro_year-2}-{hydro_year-1} not found")

    sm.data = np.where(mask, sm.data, np.nan)
    sm      = filter_closer_to_catchment(sm, catchment_poly, plot=False)

    times     = np.array([cell.point for cell in sm.coord('time').cells()])
    day_index = {(c.year, c.month, c.day): i for i, c in enumerate(times)}

    return SoilMoisture(cube=sm, data=sm.data, times=times, day_index=day_index)

def plot_sm_check(sm_data, sm_time_arr, sm_day_idx, anchor_key, yi, xi,
                  values, time_coord, n=10):
    idx = sm_day_idx[anchor_key]
    lo, hi = max(idx - n, 0), min(idx + n + 1, len(sm_time_arr))
    offsets = np.arange(lo, hi) - idx
    sm_series = sm_data[lo:hi, yi, xi]

    # daily rainfall at the peak cell (only covers the event window)
    rain_by_day = {}
    for p, v in zip(time_coord.points, values):
        d = time_coord.units.num2date(p)
        k = (d.year, d.month, d.day)
        rain_by_day[k] = rain_by_day.get(k, 0) + (0 if np.isnan(v) else v)
    rain = [rain_by_day.get((c.year, c.month, c.day), 0) for c in sm_time_arr[lo:hi]]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(offsets, sm_series, 'o-', label='SM at cell')
    ax.axvline(0,  color='k', ls='--', label='first wet day')
    ax.axvline(-1, color='r', ls='--', label='day used')
    ax.set_xlabel('days from first wet day'); ax.set_ylabel('SM')
    ax2 = ax.twinx(); ax2.bar(offsets, rain, alpha=0.3, color='tab:blue')
    ax2.set_ylabel('rain at cell (event window)')
    ax.legend(); plt.show()
    
def extract_at_cell(yi, xi, event_cube, flood_numpy, event_idx, sm, prefix, WET_THRESH):
    """Rainfall, onset-anchored SM and flood values at one cell. Keys start with `prefix`."""
    out = {f'{prefix}_y_idx': yi, f'{prefix}_x_idx': xi}

    series = np.ma.filled(event_cube.data[:, yi, xi], np.nan)
    out[f'{prefix}_rain_series'] = series.astype(np.float32)
    out[f'{prefix}_max_precip']   = float(np.nanmax(series))
    out[f'{prefix}_total_precip'] = float(np.nansum(series))

    onset = find_wet_onset(series, event_cube.coord('time'), WET_THRESH)
    if onset is None:
        out[f'{prefix}_no_wet_hours'] = True
        out[f'{prefix}_sm_day_before_onset'] = np.nan
    else:
        first, last, anchor = onset
        out[f'{prefix}_no_wet_hours'] = False
        out[f'{prefix}_first_wet_idx'] = first
        out[f'{prefix}_trimmed_duration_hrs'] = last - first + 1
        sm_field = sm_day_before(sm.data, sm.day_index, anchor)
        out[f'{prefix}_sm_day_before_onset'] = float(sm_field[yi, xi])

    for depth, d in flood_numpy.items():
        out[f'{prefix}_{depth}cm_area']   = float(d['area'][event_idx, yi, xi])
        out[f'{prefix}_{depth}cm_volume'] = int(d['vol'][event_idx, yi, xi])
    return out    


def get_rainfall_data_from_climate_data(RAIN_CUBE_ROOT, hydro_year, ens_num, row, mask, poly):

    rainfall_cube_dir = RAIN_CUBE_ROOT + f"{ens_num}/"

    # -------------------------
    # Get rainfall data for the hydro-year this event happened in
    # -------------------------
    year_cube = get_rainfall_cube_subsection(hydro_year, ens_num, rainfall_cube_dir, 
                                         row.start_indices-1,row.stop_indices)
    year_cube.data = np.where(mask, year_cube.data, np.nan)

    # -------------------------
    # Get times
    # -------------------------       
    time_coord  = year_cube.coord('time')
    t0 = time_coord.units.num2date(np.floor(time_coord.points[0]))
    t_end = time_coord.units.num2date(time_coord.points[-1])
    event_start = cftime.Datetime360Day(row.start_year, row.start_month, row.start_day, row.start_hour)

    # Check timings make sense
    event_start_within_cube = t0 <= event_start <= t_end

    if not event_start_within_cube:
        time_mismatch = True
        print("TIME MISMACH")
    else:
        time_mismatch = False     

    return year_cube, time_coord, time_mismatch

def add_rainfall_peak_time_vars(time_coord, result):
   # Get rainfall peak start times
    t = time_coord.units.num2date(time_coord.points[result['rainpeak_t_local']])
    result['rainfall_peak_day'] = cftime.Datetime360Day(t.year, t.month, t.day, t.hour)
    result['rainfall_peak_day_hydro_360day'] = ((result['rainfall_peak_day'].month % 12) * 30 + result['rainfall_peak_day'].day)
    result['rainfall_peak_day_360day'] = ((result['rainfall_peak_day'].month - 1) * 30 + result['rainfall_peak_day'].day)
    result['t_global'] =  int(result['rainpeak_t_local'] + result['start_idx']) 
   
    return result

def get_rainfall_at_peak_cells(pk, prefix, year_cube, flood_numpy, sm, x_offset, y_offset,
                               event_idx, WET_THRESH, result=None):
    if result is None:
        result = {}

    # No peak cell for this anchor (e.g. no flooding anywhere): write NaN placeholders so every
    # event has the same columns, then return. The event itself is kept.
    if pk is None:
        result[f'{prefix}_found']      = False
        result[f'{prefix}_peak_value'] = np.nan
        result[f'{prefix}_x_global']   = np.nan
        result[f'{prefix}_y_global']   = np.nan
        result[f'{prefix}_x_coord']    = np.nan
        result[f'{prefix}_y_coord']    = np.nan
        return result

    result[f'{prefix}_found']      = True
    result[f'{prefix}_peak_value'] = pk['peak_value']
    result[f'{prefix}_x_global']   = pk['x_idx'] + x_offset
    result[f'{prefix}_y_global']   = pk['y_idx'] + y_offset
    result[f'{prefix}_x_coord']    = float(year_cube.coord('projection_x_coordinate').points[pk['x_idx']])
    result[f'{prefix}_y_coord']    = float(year_cube.coord('projection_y_coordinate').points[pk['y_idx']])

    if prefix == 'rainpeak':
        result[f'{prefix}_t_local'] = pk['t_local']

    result.update(extract_at_cell(pk['y_idx'], pk['x_idx'], year_cube, flood_numpy,
                                  event_idx, sm, prefix, WET_THRESH))
    return result              


def find_peak_cell(field, min_value=0.0):
    """Peak of a (y, x) or (t, y, x) array. Returns None if there is no peak above min_value."""
    arr = np.array(np.ma.filled(field, np.nan), dtype=float)   # copy, so the cube is untouched
    arr[arr >= 1e19] = np.nan                                  # fill values

    if np.all(np.isnan(arr)) or np.nanmax(arr) <= min_value:
        return None

    idx = np.unravel_index(np.nanargmax(arr), arr.shape)
    out = {'y_idx': int(idx[-2]), 'x_idx': int(idx[-1]), 'peak_value': float(arr[idx])}
    if arr.ndim == 3:
        out['t_local'] = int(idx[0])
    return out

def find_spatial_stats_around_peak(this_event_results,year_cube,flood_numpy,flood_cubes,CATCHMENT_POLY,boundary_gdf,
                       neighbourhood_size,threshold_levels,plot=False):


    # ── Pull out the essentials for this event ──────────────────────────────
    peak_value = this_event_results['rainpeak_peak_value']
    t_local    = int(this_event_results['rainpeak_t_local'])
    yi         = int(this_event_results['rainpeak_y_idx'])   # peak cell row
    xi         = int(this_event_results['rainpeak_x_idx'])   # peak cell col

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
    flood_arrays = {(depth, var): flood_numpy[depth][var][this_event_results['event_num'] - 1, :, :]
        for depth in [10, 30]
        for var in ['area', 'vol']}

    # ══════════════════════════════════════════════════════════════════════
    # SCALE 1: Neighbourhood — fixed-size window around the peak cell
    # ══════════════════════════════════════════════════════════════════════
    n  = neighbourhood_size
    y0 = max(0, yi - n);  y1 = min(rain.shape[0], yi + n + 1)
    x0 = max(0, xi - n);  x1 = min(rain.shape[1], xi + n + 1)
    neighbourhood = rain[y0:y1, x0:x1]

#     results["neighbourhood_rain_total"] = round(float(np.nansum(neighbourhood)),1)
#     results["neighbourhood_rain_sum_excl_peak"] = round(float(np.nansum(neighbourhood) - peak_value),1)
#     results["neighbourhood_n_cells"] = round(int(np.sum(~np.isnan(neighbourhood))),1)

    results = {
        # Total rainfall summed across the neighbourhood window
        'neighbourhood_rain_total': float(np.nansum(neighbourhood)),
        # Same, but excluding the peak cell's own value (isolates the
        # contribution of the surrounding cells only)
        'neighbourhood_rain_sum_excl_peak': float(np.nansum(neighbourhood) - peak_value),
        # How many valid (non-NaN, i.e. in-catchment) cells the window
        # actually covered — window can be clipped near catchment edges
        'neighbourhood_n_cells': int(np.sum(~np.isnan(neighbourhood)))}
    
    # Flood outcomes summed over the same neighbourhood window
    for depth in [10, 30]:
        for var in ['area', 'vol']:
            results[f'neighbourhood_flood_{depth}_{var}'] = float(np.nansum(flood_arrays[(depth, var)][y0:y1, x0:x1]))


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
    
    this_event_results.update(results)
    
    return this_event_results
