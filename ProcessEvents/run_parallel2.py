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
import subprocess
import getpass
import signal
import sys
import logging

from functions import *
from functions_stage2 import get_rainfall_cube_subsection

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

## ------------------------------------------------------------ ###
# Define which catchments to run
## ------------------------------------------------------------ ###
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

# ── Logging setup ─────────────────────────────────────────────────────────────
def setup_worker_logger(catchment_num):
    """Each worker logs to its own file so output never interleaves."""
    log_path = os.path.join(OUT_DIR, f"logs/catchment_{catchment_num}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(f"catchment_{catchment_num}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    return logger


# ── Safety checks ─────────────────────────────────────────────────────────────
def check_no_existing_workers():
    """
    Warn if leftover Python processes from a previous run are still alive.
    These compete for CPU/memory and can make everything run much slower.
    """
    user   = getpass.getuser()
    result = subprocess.run(['pgrep', '-u', user, '-f', 'python'],
                            capture_output=True, text=True)
    pids     = [p for p in result.stdout.strip().split('\n') if p]
    n_other  = len(pids) - 1  # subtract current process

    if n_other > 0:
        print(f"\nWARNING: {n_other} existing Python processes found.")
        print("These may be leftover workers from a previous run and will slow things down.")
        print(f"To clean up run: pkill -u {user} python")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Clean up old processes and re-run.")
            sys.exit(0)
    else:
        print("✓ No leftover processes found — clean to start.")


# ── Main processing function ──────────────────────────────────────────────────
def process_catchment(catchment_num, verbose=False):
    """
    verbose=True  → prints directly to console/notebook output
    verbose=False → writes to log file only (for use with ProcessPoolExecutor)
    """

    # ── Set up output routing ─────────────────────────────────────────────────
    # When running in parallel, verbose=False routes all output to a per-catchment
    # log file so workers don't interleave. When debugging in a notebook,
    # verbose=True prints directly so you see output immediately.
    log = setup_worker_logger(catchment_num)

    def emit(msg):
        """Single call routes to print or log depending on verbose flag."""
        if verbose:
            print(msg)
        else:
            log.info(msg)

    catchment_name  = CATCHMENT_LOOKUP_DICT[str(catchment_num)]
    emit(f"Starting — {catchment_name}")
    boundary_gdf    = CATCHMENTS[CATCHMENTS['HA_NUM'] == str(catchment_num)]
    _CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]
    results         = []

    for ens_num in ENSEMBLE_MEMBERS:
        out_fp = f"{OUT_DIR}/Catchment_{catchment_num}/{catchment_name}_EM{ens_num}.pkl"
        if os.path.isfile(out_fp):
            emit(f"EM{ens_num}: already exists, skipping")
            continue

        results_this_ens = []
        rainfall_events  = pd.read_csv(RAINFALL_CSV_DIR + f"{catchment_name}_{ens_num}_full_events_with_event_nums.csv")
        rainfall_events['event_num'] = range(1, len(rainfall_events) + 1)
        rainfall_cube_dir = f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens_num}/"
        emit(f"EM{ens_num}: {len(rainfall_events)} events to process")

        event_details_cache = {
            ev: get_rainfall_event_details(rainfall_events, ev)
            for ev in rainfall_events['event_num']}

        t0 = time.time()
        full_rain_cube = get_rainfall_cube_subsection(2015, ens_num, rainfall_cube_dir,1,2)
        FULL_MASK_2D   = mask_cube_with_catchment_full_grid(full_rain_cube[0], _CATCHMENT_POLY, method='full_cell')
        emit(f"EM{ens_num}: mask created in {time.time()-t0:.1f}s")
        del full_rain_cube

        for year, events_in_year in rainfall_events.groupby('start_year'):
            t0 = time.time()
            year_cube = get_rainfall_cube(year, ens_num, rainfall_cube_dir)
            year_cube, x_offset, y_offset = subset_cube_to_bbox(year_cube, _CATCHMENT_POLY, buffer=0)
            #emit(f"Get cube: {time.time()-t0:.1f}s")

            nx_sub      = year_cube.shape[2]
            ny_sub      = year_cube.shape[1]
            mask_2d_sub = FULL_MASK_2D[y_offset:y_offset+ny_sub, x_offset:x_offset+nx_sub]
            time_coord  = year_cube.coord('time')

            for row in events_in_year.itertuples():
                event_num     = int(row.event_num)
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

            emit(f"EM{ens_num} | year {year}: {len(events_in_year)} events in {time.time()-t0:.1f}s")
            del year_cube
            gc.collect()

        pd.DataFrame(results_this_ens).to_pickle(
            os.path.join(OUT_DIR, f"Catchment_{catchment_num}", f"{catchment_name}_EM{ens_num}.pkl"))
        emit(f"EM{ens_num}: saved")

    results_df  = pd.DataFrame(results)
    combined_fp = os.path.join(OUT_DIR, f"Catchment_{catchment_num}", f"{catchment_name}.pkl")
    results_df.to_pickle(combined_fp)
    emit(f"Combined file saved: {combined_fp}")

    for ens_num in ENSEMBLE_MEMBERS:
        fp = os.path.join(OUT_DIR, f"Catchment_{catchment_num}", f"{catchment_name}_EM{ens_num}.pkl")
        if os.path.exists(fp):
            os.remove(fp)
    emit("Done — individual EM files cleaned up")

    return catchment_num



# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Check for leftover processes before starting ──────────────────────────
    check_no_existing_workers()

    # ── Signal handler so Ctrl+C / SIGTERM kills workers cleanly ─────────────
    # Without this, interrupting the script leaves worker processes alive,
    # which then compete with the next run and cause significant slowdown.
    executor = None

    def shutdown_handler(sig, frame):
        print(f"\nSignal {sig} received — shutting down workers cleanly...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        print("Workers shut down. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    N_WORKERS = 2  # tune to your node's CPU and memory

    print(f"\nStarting parallel run: {len(catchments_to_run_sorted)} catchments, "
          f"{N_WORKERS} workers")
    print(f"Logs: {OUT_DIR}logs/catchment_<num>.log\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        executor = ex  # expose to signal handler
        futures  = {ex.submit(process_catchment, cn, verbose=True): cn
                    for cn in catchments_to_run_sorted}

        for future in as_completed(futures):
            cn = futures[future]
            try:
                future.result()
                print(f"✓ Catchment {cn} complete")
            except Exception as e:
                print(f"✗ Catchment {cn} failed: {e} — check logs/catchment_{cn}.log")

    print("\nAll catchments complete.")