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

from functions import *


import logging

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

# ── Config ────────────────────────────────────────────────────────────────────
MOLLY_DIR_FF     = "/scratch/hydro4/users/kv25483/FutureFlood/"
RAINFALL_CSV_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall_events/fixed_threshold_30mm_with_volume/"
OUT_DIR          = "/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/"

from config import CATCHMENT_LOOKUP_DICT, OUT_DIR, CATCHMENTS, ENSEMBLE_MEMBERS # MOLLY_DIR_FF, RAINFALL_CSV_DIR, , ENSEMBLE_MEMBERS,  

### ------------------------------------------------------------ ###
# Define which catchments to run
### ------------------------------------------------------------ ###
all_catchments = set(CATCHMENT_LOOKUP_DICT.keys())
files = os.listdir(OUT_DIR)
completed_catchments = {re.search(r'Catchment_(.+)', f).group(1) for f in files if re.search(r'Catchment_(.+)', f)}
catchments_to_run = (all_catchments - completed_catchments) - {105} - {14}

catchments_with_flood_output = []
## Loop through all catchments
for catchment_num in all_catchments:
    catchment_name = CATCHMENT_LOOKUP_DICT[catchment_num]
    fp = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/Catchment_{catchment_num}/{catchment_name}.pkl"
    flood_fp = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{catchment_num}/Ens01_{catchment_num}/10cm/flooded_area_5km_total_Ens01_{catchment_num}_10cm.nc"
        
    if os.path.isfile(flood_fp):
        catchments_with_flood_output.append(catchment_num)

# ── Check which (catchment, ens) combinations still need processing ───────────
# Assumes outputs are saved as: OUT_DIR/Catchment_{num}/Ens_{ens}.pkl
# If your output structure is different, adjust the path accordingly.

missing = {}  # {catchment_num: [list of missing ens]}
to_prioritise ={}

for catchment_num in catchments_with_flood_output:
    catchment_name = CATCHMENT_LOOKUP_DICT[catchment_num]

    # Load just enough to know which ens members exist for this catchment
    all_ens = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']

    overall_fp = out_fp = f"{OUT_DIR}/Catchment_{catchment_num}/{catchment_name}.pkl"
    if os.path.isfile(overall_fp):
        print(f"Catchment {catchment_num} ({catchment_name}) is complete")
        pass
    else:
    
        missing_ens = []
        for ens in sorted(all_ens):
            out_fp = f"{OUT_DIR}/Catchment_{catchment_num}/{catchment_name}_EM{ens}.pkl"
            if not os.path.isfile(out_fp):
                missing_ens.append(ens)

        if missing_ens:
            missing[catchment_num] = missing_ens
            print(f"Catchment {catchment_num} ({catchment_name}): missing ens {missing_ens}")
        else:
            print(f"Catchment {catchment_num} ({catchment_name}): complete")

# Summary
total_missing = sum(len(v) for v in missing.values())
print(f"\n{len(missing)} catchments incomplete, {total_missing} (catchment, ens) jobs remaining")        
        
# Sort catchments by number of missing ens (ascending) so we tackle the
# nearly-complete ones first — minimises total remaining work to get results.
sorted_missing = sorted(missing.items(), key=lambda x: len(x[1]))

print("\nCatchments sorted by missing ens (least first):")
print(f"{'Catchment':<15} {'Name':<30} {'Missing ens':<15} {'Ens list'}")
print("-" * 80)
for catchment_num, missing_ens in sorted_missing:
    catchment_name = CATCHMENT_LOOKUP_DICT[catchment_num]
    print(f"{catchment_num:<15} {catchment_name:<30} {len(missing_ens):<15} {missing_ens}")

# Also gives you a clean list to feed directly into your processing loop
catchments_to_run_sorted = [catchment_num for catchment_num, _ in sorted_missing]

# def process_catchment(catchment_num):
#     """Processes a single catchment across all ensemble members."""
#     catchment_name   = CATCHMENT_LOOKUP_DICT[str(catchment_num)]
#     catchment_out    = os.path.join(OUT_DIR, f"Catchment_{catchment_num}")
#     os.makedirs(catchment_out, exist_ok=True)

#     # Prepare spatial data for this catchment and ensemble member
#     boundary_gdf    = CATCHMENTS[CATCHMENTS['HA_NUM'] == str(catchment_num)]
#     _CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]

#     # Create a list to store ALL results (all ensemble members)
#     results = []
    
#     # ── Loop over ensemble members, running main script ──────────────────────────
#     for ens_num in ENSEMBLE_MEMBERS:
#         print(ens_num)
#         if not os.path.isfile(f"{OUT_DIR}/Catchment_{catchment_num}/{catchment_name}_EM{ens_num}.pkl"):
#             # Gets rid of things that no longer need to be stored
#             gc.collect()

#             # Create a list to store results  for this ensemble member
#             results_this_ens = []

#             # Get a dataframe containing details of extreme rainfall events
#             rainfall_events = pd.read_csv(RAINFALL_CSV_DIR + f"{catchment_name}_{ens_num}_full_events_with_event_nums.csv")
#             # Add variable giving each event a number
#             rainfall_events['event_num'] = range(1, len(rainfall_events) + 1)

#             # Establish filepath where netCDF of rainfall data lives for this ensemble member
#             rainfall_cube_dir = f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens_num}/"

#             # Print initialisation statement (now we know the number of events) 
#             print(f"Running for {catchment_name}, for {len(rainfall_events)} events for EM: {ens_num}")

#             # ---- CACHE EVENT DETAILS (avoid recomputation) ----
#             # Create a dictionary that stores the output of pre-computing and storing the output of get_rainfall_event_details() 
#             # for every event, keyed by event number. 
#             event_details_cache = {
#                 ev: get_rainfall_event_details(rainfall_events, ev)
#                 for ev in rainfall_events['event_num']}

#             # Create a mask which masks out any cells not within the catchment
#             # This is year agnostic, and is applicable to all years
#             # Has shape 244, 180 (full GB)
#             # Create mask for whole country, and then trim it to the same extent as the rainfall cube before applying it
#             # This allows creating mask just once, and then applying for each year
#             start_time1 = time.time()
#             full_rain_cube = get_rainfall_cube(2015, ens_num, rainfall_cube_dir)   # any year, grid is same
#             FULL_MASK_2D = mask_cube_with_catchment_full_grid(full_rain_cube[0], _CATCHMENT_POLY, method='full_cell')
#             print(f"Creating a mask for EM{ens_num} in {round(time.time() - start_time1, 2)}s")    

#             # ---- PROCESS BY YEAR ----
#             for year, events_in_year in rainfall_events.groupby('start_year'):
#                 start_time_total = time.time()
#                 print(f"\nProcessing year {year} with {len(events_in_year)} events")

#                 # ---- LOAD CUBE for this year and subset to catchment boundaries ----
#                 start_time1 = time.time()
#                 year_cube = get_rainfall_cube(year, ens_num, rainfall_cube_dir)
#                 year_cube, x_offset, y_offset = subset_cube_to_bbox(year_cube, _CATCHMENT_POLY, buffer=0)
#                 print(f"Loaded cube for {year} in {round(time.time() - start_time1, 2)}s")

#                 # ---- Create version of mask for this year and trimmed to catchment
#                 nx_sub = year_cube.shape[2]
#                 ny_sub = year_cube.shape[1]
#                 mask_2d_sub = FULL_MASK_2D[y_offset:y_offset+ny_sub, x_offset:x_offset+nx_sub]

#                 # Cache time coordinate once
#                 time_coord = year_cube.coord('time')

#                 # ---- PROCESS for each EVENT in this year ----
#                 for row in events_in_year.itertuples():

#                     # Get event details for this event
#                     event_num = int(row.event_num)
#                     event_details = event_details_cache[event_num]
#                     #  print(f"max precip from csv: {event_details['max_precip_from_csv']}")
#                     # print(f"max precip from cube: {np.nanmax(year_cube[event_details['start_idx']:event_details['stop_idx'],:,:].data)}")
#                     #print(f"Ens: {ens_num}, event num: {event_num}, event year: {year}")
#                     print(f"Event num: {event_num}")

#                     # Returns the location in time and space, and the value, of the maximum precipitation value
#                     # x_idx, y_idx records its location in the subsetted cube (subsetted to the catchment boundaries)
#                     # x_idx_global, y_idx_global records its location in the whole UK cube
#                     this_event_results = find_max_precip_location(year_cube, event_details['start_idx'], event_details['stop_idx'],
#                         x_offset = x_offset, y_offset = y_offset, mask_2d=mask_2d_sub)

#                     # Metadata
#                     this_event_results['ens'] = ens_num
#                     this_event_results['start_idx'] = event_details['start_idx']
#                     this_event_results['stop_idx'] = event_details['stop_idx']

#                     # Peak day
#                     t = time_coord.units.num2date(
#                         time_coord.points[this_event_results['t_global']])
#                     this_event_results['rainfall_peak_day'] = cftime.Datetime360Day(t.year, t.month, t.day)

#                     # Rainall_at_peak is a 1D cube, containing the rainfall data for a full year at the grid cell where the maximum occurred
#                     # The find_temporal_profile function then searches this 1D cube and extracts rainfall between the start and stop index
#                     # These rainfall values and times are saved and then also a number of temporal profile variables are calculated
#                     # These are added to the overall results dictionary
#                     rainfall_at_peak = get_data_at_peak_cell(year_cube, this_event_results, 'x_idx', 'y_idx')
#                     temp_profile_dict = find_temporal_profile(rainfall_at_peak, this_event_results, plot=False)
#                     this_event_results = {**this_event_results, **temp_profile_dict}

#                     # Add to the overall list of results and the results for one EM
#                     results.append(this_event_results)
#                     results_this_ens.append(this_event_results)

#                 print(f"Completed operation for EM {ens_num}, {year} in {round(time.time() - start_time_total, 2)}s")

#                 # Save the results for this EM
#                 results_this_ens_df = pd.DataFrame(results_this_ens)
#                 # results_this_ens_df.to_pickle(os.path.join(catchment_out, f"{catchment_name}_EM{ens_num}.pkl"))

#                 # ---- CLEAN UP MEMORY ----
#                 del year_cube
#                 gc.collect()
#         else:
#             print(f"EM{ens_num} already exists, skipping")
            
#     # Covnert overall results to dataframe and save to a pickle file
#     results_df = pd.DataFrame(results)
#     results_df.to_pickle(os.path.join(catchment_out, f"{catchment_name}.pkl"))
    
#     # Clean up per-ensemble files
#     if os.path.exists(os.path.join(catchment_out, f"{catchment_name}.pkl")):
#         print("Produced overall output")
#         for ens_num in ENSEMBLE_MEMBERS:
#             output_fp = os.path.join(catchment_out, f"{catchment_name}_EM{ens_num}.pkl")
#             os.remove(output_fp)
#             print(f"Deleted {output_fp}")
#     else:
#         print("WARNING: Combined file not found — individual ensemble files kept as fallback.")

#     return catchment_num  # signals success

def process_catchment(catchment_num):
    log = setup_worker_logger(catchment_num)
    catchment_name = CATCHMENT_LOOKUP_DICT[str(catchment_num)]
    log.info(f"Starting — {catchment_name}")

    boundary_gdf    = CATCHMENTS[CATCHMENTS['HA_NUM'] == str(catchment_num)]
    _CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]
    results         = []

    for ens_num in ENSEMBLE_MEMBERS:
        out_fp = f"{OUT_DIR}/Catchment_{catchment_num}/{catchment_name}_EM{ens_num}.pkl"
        if os.path.isfile(out_fp):
            log.info(f"EM{ens_num}: already exists, skipping")
            continue

        results_this_ens = []
        rainfall_events  = pd.read_csv(RAINFALL_CSV_DIR + f"{catchment_name}_{ens_num}_full_events_with_event_nums.csv")
        rainfall_events['event_num'] = range(1, len(rainfall_events) + 1)
        rainfall_cube_dir = f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens_num}/"
        log.info(f"EM{ens_num}: {len(rainfall_events)} events to process")

        event_details_cache = {
            ev: get_rainfall_event_details(rainfall_events, ev)
            for ev in rainfall_events['event_num']}

        t0 = time.time()
        full_rain_cube = get_rainfall_cube(2015, ens_num, rainfall_cube_dir)
        FULL_MASK_2D   = mask_cube_with_catchment_full_grid(full_rain_cube[0], _CATCHMENT_POLY, method='full_cell')
        log.info(f"EM{ens_num}: mask created in {time.time()-t0:.1f}s")

        for year, events_in_year in rainfall_events.groupby('start_year'):
            t0 = time.time()
            year_cube, x_offset, y_offset = subset_cube_to_bbox(
                get_rainfall_cube(year, ens_num, rainfall_cube_dir), _CATCHMENT_POLY, buffer=0)
            
            nx_sub   = year_cube.shape[2]
            ny_sub   = year_cube.shape[1]
            mask_2d_sub = FULL_MASK_2D[y_offset:y_offset+ny_sub, x_offset:x_offset+nx_sub]
            time_coord  = year_cube.coord('time')

            for row in events_in_year.itertuples():
                event_num    = int(row.event_num)
                event_details = event_details_cache[event_num]

                this_event_results = find_max_precip_location(
                    year_cube, event_details['start_idx'], event_details['stop_idx'],
                    x_offset=x_offset, y_offset=y_offset, mask_2d=mask_2d_sub)

                this_event_results['ens']       = ens_num
                this_event_results['start_idx'] = event_details['start_idx']
                this_event_results['stop_idx']  = event_details['stop_idx']

                t = time_coord.units.num2date(time_coord.points[this_event_results['t_global']])
                this_event_results['rainfall_peak_day'] = cftime.Datetime360Day(t.year, t.month, t.day)

                rainfall_at_peak  = get_data_at_peak_cell(year_cube, this_event_results, 'x_idx', 'y_idx')
                temp_profile_dict = find_temporal_profile(rainfall_at_peak, this_event_results, plot=False)
                this_event_results = {**this_event_results, **temp_profile_dict}

                results.append(this_event_results)
                results_this_ens.append(this_event_results)

            log.info(f"EM{ens_num} | year {year}: {len(events_in_year)} events in {time.time()-t0:.1f}s")
            del year_cube
            gc.collect()

        pd.DataFrame(results_this_ens).to_pickle(
            os.path.join(OUT_DIR, f"Catchment_{catchment_num}", f"{catchment_name}_EM{ens_num}.pkl"))
        log.info(f"EM{ens_num}: saved")

    # ── Combine and clean up ──────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    combined_fp = os.path.join(OUT_DIR, f"Catchment_{catchment_num}", f"{catchment_name}.pkl")
    results_df.to_pickle(combined_fp)
    log.info(f"Combined file saved: {combined_fp}")

    for ens_num in ENSEMBLE_MEMBERS:
        fp = os.path.join(OUT_DIR, f"Catchment_{catchment_num}", f"{catchment_name}_EM{ens_num}.pkl")
        if os.path.exists(fp):
            os.remove(fp)
    log.info("Done — individual EM files cleaned up")

    return catchment_num



# if __name__ == "__main__":
#     N_WORKERS = 8  # tune to your node's CPU/memory — each worker loads full cubes

#     with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
#         futures = {executor.submit(process_catchment, cn): cn for cn in catchments_to_run_sorted}
#         for future in as_completed(futures):
#             cn = futures[future]
#             try:
#                 future.result()
#                 print(f"✓ Catchment {cn} complete")
#             except Exception as e:
#                 print(f"✗ Catchment {cn} failed: {e}")

if __name__ == "__main__":
    N_WORKERS = 8

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_catchment, cn): cn for cn in catchments_to_run_sorted}
        for future in as_completed(futures):
            cn = futures[future]
            try:
                future.result()
                # Only high-level status goes to console — details are in the log file
                print(f"✓ Catchment {cn} complete — see logs/catchment_{cn}.log")
            except Exception as e:
                print(f"✗ Catchment {cn} FAILED: {e} — see logs/catchment_{cn}.log")