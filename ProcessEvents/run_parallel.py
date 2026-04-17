import numpy as np
import pandas as pd
import cftime
import geopandas as gpd
import re
import time
import gc
import os
import shapely
from concurrent.futures import ProcessPoolExecutor, as_completed

from functions_clean import *

# ── Config ────────────────────────────────────────────────────────────────────
MOLLY_DIR_FF     = "/scratch/hydro4/users/kv25483/FutureFlood/"
RAINFALL_CSV_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall_events/fixed_threshold_30mm_with_volume/"
OUT_DIR          = "/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/"

catchments       = gpd.read_file(MOLLY_DIR_FF + "Data/CatchmentShapefiles/hyd_areas_GB_with_subcatchments.shp")
catchment_lookup = pd.read_csv("/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/CEH_catchments/CEH_IHU_with_coastline/hyd_areas_GB_with_subcatchments_no_spaces.csv")
catchment_lookup_dict = dict(zip(catchment_lookup["HA_NUM"], catchment_lookup["HA_NAME"]))

ENSEMBLE_MEMBERS = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']

all_catchments = set(catchment_lookup_dict.keys())
files = os.listdir(OUT_DIR)
completed_catchments = {re.search(r'Catchment_(.+)', f).group(1) for f in files if re.search(r'Catchment_(.+)', f)}
catchments_to_run = (all_catchments - completed_catchments) - {105} - {14}


def process_catchment(catchment_num):
    """Processes a single catchment across all ensemble members."""
    catchment_name = catchment_lookup_dict[str(catchment_num)]
    catchment_out  = os.path.join(OUT_DIR, f"Catchment_{catchment_num}")
    os.makedirs(catchment_out, exist_ok=True)

    boundary_gdf    = catchments[catchments['HA_NUM'] == str(catchment_num)]
    _CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]

    results = []

    for ens_num in ENSEMBLE_MEMBERS:
        results_this_ens = []
        gc.collect()

        rainfall_events = pd.read_csv(
            RAINFALL_CSV_DIR + f"{catchment_name}_{ens_num}_full_events_with_event_nums.csv")

        rainfall_cube_dir = f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens_num}/"
        rainfall_events['event_num'] = range(1, len(rainfall_events) + 1)

        print(f"[{catchment_num}] Running {catchment_name}, {len(rainfall_events)} events, EM: {ens_num}")

        event_details_cache = {
            ev: get_rainfall_event_details(rainfall_events, ev)
            for ev in rainfall_events['event_num']}

        rainfall_events['year'] = rainfall_events['event_num'].map(lambda ev: event_details_cache[ev]['yr'])

        full_rain_cube = get_rainfall_cube(2015, ens_num, rainfall_cube_dir)
        FULL_MASK_2D   = mask_cube_with_catchment_full_grid(full_rain_cube[0], _CATCHMENT_POLY, method='full_cell')

        for year, events_in_year in rainfall_events.groupby('year'):
            start_time_total = time.time()
            print(f"[{catchment_num}] Year {year}, {len(events_in_year)} events")

            year_cube = get_rainfall_cube(year, ens_num, rainfall_cube_dir)
            year_cube, x_offset, y_offset = subset_cube_to_bbox(year_cube, _CATCHMENT_POLY, buffer=0)

            nx_sub    = year_cube.shape[2]
            ny_sub    = year_cube.shape[1]
            mask_2d_sub = FULL_MASK_2D[y_offset:y_offset+ny_sub, x_offset:x_offset+nx_sub]
            time_coord  = year_cube.coord('time')

            for row in events_in_year.itertuples():
                event_num    = int(row.event_num)
                event_details = event_details_cache[event_num]

                this_event_results = find_max_precip_location(
                    year_cube, event_details['start_idx'], event_details['stop_idx'],
                    x_offset=x_offset, y_offset=y_offset, mask_2d=mask_2d_sub)

                this_event_results['ens']       = ens_num
                this_event_results['year']      = year
                this_event_results['start_idx'] = event_details['start_idx']
                this_event_results['stop_idx']  = event_details['stop_idx']

                t = time_coord.units.num2date(time_coord.points[this_event_results['t_global']])
                this_event_results['rainfall_peak_day'] = cftime.Datetime360Day(t.year, t.month, t.day)

                rainfall_at_peak  = get_data_at_peak_cell(year_cube, this_event_results)
                temp_profile_dict = find_temporal_profile(rainfall_at_peak, this_event_results, plot=False)
                this_event_results = {**this_event_results, **temp_profile_dict}

                results.append(this_event_results)
                results_this_ens.append(this_event_results)

            print(f"[{catchment_num}] EM {ens_num}, {year} done in {round(time.time() - start_time_total, 2)}s")

            results_this_ens_df = pd.DataFrame(results_this_ens)
            results_this_ens_df.to_pickle(os.path.join(catchment_out, f"{catchment_name}_EM{ens_num}.pkl"))

            del year_cube
            gc.collect()

    results_df = pd.DataFrame(results)
    results_df.to_pickle(os.path.join(catchment_out, f"{catchment_name}.pkl"))

    if os.path.exists(os.path.join(catchment_out, f"{catchment_name}.pkl")):
        print(f"[{catchment_num}] Produced overall output")
        for ens_num in ENSEMBLE_MEMBERS:
            fp = os.path.join(catchment_out, f"{catchment_name}_EM{ens_num}.pkl")
            if os.path.exists(fp):
                os.remove(fp)
    else:
        print(f"[{catchment_num}] WARNING: Combined file not found — keeping ensemble files.")

    return catchment_num  # signals success


if __name__ == "__main__":
    N_WORKERS = 4  # tune to your node's CPU/memory — each worker loads full cubes

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_catchment, cn): cn for cn in catchments_to_run}
        for future in as_completed(futures):
            cn = futures[future]
            try:
                future.result()
                print(f"✓ Catchment {cn} complete")
            except Exception as e:
                print(f"✗ Catchment {cn} failed: {e}")