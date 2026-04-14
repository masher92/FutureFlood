import geopandas as gpd
import pandas as pd
import iris
import cftime
from iris.warnings import IrisCfMissingVarWarning
import warnings
warnings.filterwarnings("ignore", category=IrisCfMissingVarWarning)
iris.FUTURE.date_microseconds = True
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import re

from functions import * # process_events, prepare_flood_cube,load_3d_cube

# ── Config ────────────────────────────────────────────────────────────────────
RAINFALL_CSV_DIR   = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall_events/fixed_threshold_30mm_with_volume/" 
MOLLY_DIR_FF = "/scratch/hydro4/users/kv25483/FutureFlood/"
METHOD          = 'center_point'
OUT_DIR = '/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/'

catchment_lookup = pd.read_csv("/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/CEH_catchments/CEH_IHU_with_coastline/hyd_areas_GB_with_subcatchments_no_spaces.csv")
catchment_lookup_dict = dict(zip(catchment_lookup["HA_NUM"],catchment_lookup["HA_NAME"]))

CATCHMENT_NUM = 102
CATCHMENT_NAME = catchment_lookup_dict[str(CATCHMENT_NUM)]

# ── Load inputs ───────────────────────────────────────────────────────────────
catchments = gpd.read_file(MOLLY_DIR_FF + "Data/CatchmentShapefiles/hyd_areas_GB_with_subcatchments.shp")
boundary_gdf = catchments[catchments['HA_NUM'] == str(CATCHMENT_NUM)]
boundary_gdf.reset_index(inplace=True, drop=True)
catchment_poly = boundary_gdf.geometry[0]

# Per-worker global caches
FLOOD_CUBES = None
RAIN_CUBES = {}
SM_CUBES = {}
CATCHMENT_POLY = None

ENSEMBLE_MEMBERS = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']


# -----------------------------
# FILTER TO ONLY INCOMPLETE CATCHMENTS
# -----------------------------
def check_which_files_to_process(ha_num):
    ### check if the outputs already exist, if they do, exclude this catchment
    CATCHMENT_OUT_DIR = os.path.join(OUT_DIR, f"Catchment_{ha_num}")
    print(CATCHMENT_OUT_DIR)
    if os.path.exists(os.path.join(CATCHMENT_OUT_DIR, f"results_all_ensembles.csv")):
        return False
    
    # Check if the 5km agrgegated outputs have been made
    out_dir_base = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{ha_num}"
    for ens in ENSEMBLE_MEMBERS:
        for thr in ["10cm", "30cm"]:
            out_dir = os.path.join(out_dir_base, f"Ens{ens}_{ha_num}", thr)
            for kind in ("area", "volume"):
                fname = f"flooded_{kind}_5km_total_Ens{ens}_{ha_num}_{thr}.nc"
                if not os.path.exists(os.path.join(out_dir, fname)):
                    return False
                else:
                    print("5km outputs created already")
                    return True

def main():
    files = os.listdir("/scratch/hydro5/users/la17355/FUTURE-FLOOD/Results/Pluvial/v4/tifs/")
    
    catchment_numbers = set()
    for f in files:
        match = re.search(r'Ens15_(.+)', f)
        if match:
            catchment_numbers.add(match.group(1))
    print(catchment_numbers)
    
    catchments_to_run = {c for c in catchment_numbers if check_which_files_to_process(c)}
    print(f"{len(catchments_to_run)} catchments to process: {sorted(catchments_to_run)}")

    for CATCHMENT_NUM in sorted([104]):

        CATCHMENT_NAME = catchment_lookup_dict[str(CATCHMENT_NUM)]
        CATCHMENT_OUT_DIR = os.path.join(OUT_DIR, f"Catchment_{CATCHMENT_NUM}")
        os.makedirs(CATCHMENT_OUT_DIR, exist_ok=True)

        # Reload boundary for this catchment
        boundary_gdf = catchments[catchments['HA_NUM'] == str(CATCHMENT_NUM)]
        boundary_gdf.reset_index(inplace=True, drop=True)

        print(f"\n{'='*60}")
        print(f"Processing catchment {CATCHMENT_NUM} ({CATCHMENT_NAME})")
        print(f"{'='*60}")

        all_results = []  # reset per catchment

        for ens in tqdm(ENSEMBLE_MEMBERS, desc=f"Catchment {CATCHMENT_NUM} ensembles"):
            if not os.path.isfile(os.path.join(CATCHMENT_OUT_DIR, f"results_ens_{ens}.csv")):
                print(f"Running for EM {ens}")
                tqdm.write(f"🚀 Running ensemble {ens}")

                rainfall_cube_dir = f'/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens}/'
                flood_dir = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{CATCHMENT_NUM}/Ens{ens}_{CATCHMENT_NUM}/"
                sm_dir = f'/scratch/hydro4/shared_data/climate_projections/UKCP18/UKCP_local/Soil_moisture/5km_regridded/Ens_{ens}/'

                rainfall_events = pd.read_csv(RAINFALL_CSV_DIR + f"{CATCHMENT_NAME}_{ens}_full_events_with_event_nums.csv")
                event_nums_to_process = range(1, len(rainfall_events) + 1)
                
                event_nums_to_process = range(1, 3)

                results = process_events_parallel_fast(
                    catchment_num     = CATCHMENT_NUM,
                    event_nums        = event_nums_to_process,
                    rainfall_events   = rainfall_events,
                    rainfall_cube_dir = rainfall_cube_dir,
                    sm_dir            = sm_dir,
                    flood_dir         = flood_dir,
                    ens_num           = ens,
                    method            = METHOD,
                    n_workers         = 8)
                results["ensemble"] = ens

                # results = process_events_serial_fast(
                #     catchment_num     = CATCHMENT_NUM,
                #     event_nums        = event_nums_to_process,
                #     rainfall_events   = rainfall_events,
                #     rainfall_cube_dir = rainfall_cube_dir,
                #     sm_dir            = sm_dir,
                #     flood_dir         = flood_dir,
                #     ens_num           = ens,
                #     method            = METHOD,
                # plot_spatial=False, plot_temporal=False)

                ens_csv = os.path.join(CATCHMENT_OUT_DIR, f"results_ens_{ens}.csv")
                results.to_csv(ens_csv, index=False)
                all_results.append(results)
            else:
                print(f"Results for EM {ens} already exist, loading from file")
                results = pd.read_csv(os.path.join(CATCHMENT_OUT_DIR, f"results_ens_{ens}.csv"))
                all_results.append(results)
                      
        # Combine and save
        final_results = pd.concat(all_results, ignore_index=True)
        all_csv = os.path.join(CATCHMENT_OUT_DIR, "results_all_ensembles.csv")
        final_results.to_csv(all_csv, index=False)
        print(f"Saved combined results to {all_csv}")

        # Clean up per-ensemble files
        if os.path.exists(all_csv):
            for ens in ENSEMBLE_MEMBERS:
                ens_csv = os.path.join(CATCHMENT_OUT_DIR, f"results_ens_{ens}.csv")
                if os.path.exists(ens_csv):
                    os.remove(ens_csv)
                    print(f"Deleted {ens_csv}")
        else:
            print("WARNING: Combined file not found — individual ensemble files kept as fallback.")
            
            
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()           