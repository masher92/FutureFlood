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

from functions import * # process_events, prepare_flood_cube,load_3d_cube

# ── Config ────────────────────────────────────────────────────────────────────
# ENS_NUM = '01'
RAINFALL_CSV_DIR   = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall_events/fixed_threshold_30mm_with_volume/" 
#RAINFALL_CUBE_DIR = f'/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ENS_NUM}/'
#FLOOD_DIR = f"/scratch/hydro5/users/la17355/FUTURE-FLOOD/Results/Pluvial/v4/5km_total/Ens{ENS_NUM}_23/"
MOLLY_DIR_FF = "/scratch/hydro4/users/kv25483/FutureFlood/"
METHOD          = 'center_point'
THRESHOLD_LEVEL = 0.3

#SM_DIR = f'/scratch/hydro4/shared_data/climate_projections/UKCP18/UKCP_local/Soil_moisture/5km_regridded/Ens_{ENS_NUM}/'

# ── Load inputs ───────────────────────────────────────────────────────────────
# home_dir = "/scratch/hydro4/users/" 
#ncl_events = pd.read_csv(RAINFALL_CSV_DIR + f"Tyne(Northumberland)_{ENS_NUM}_full_events_with_event_nums.csv")
catchments = gpd.read_file(MOLLY_DIR_FF + "Data/NewcastleExample/catchment_identifier/hyd_areas_GB_with_subcatchments.shp")
boundary_gdf = catchments[catchments['HA_NUM'] == "23"]
boundary_gdf.reset_index(inplace=True, drop=True)
catchment_poly = boundary_gdf.geometry[0]

# flood_types = ['area','volume']
# depths = [10,30]
# flood_outputs = {
#     f'fld_{ft}_{d}cm': prepare_flood_cube(load_3d_cube(f"{FLOOD_DIR}{d}cm/flooded_{ft}_5km_total_Ens{ENS_NUM}_23_{d}cm.nc"), catchment_poly)
#     for ft in flood_types for d in depths}


# Per-worker global caches
FLOOD_CUBES = None
RAIN_CUBES = {}
SM_CUBES = {}
CATCHMENT_POLY = None

def init_worker(boundary_gdf, flood_dir, ens_num):
    global FLOOD_CUBES, CATCHMENT_POLY, RAIN_CUBES, SM_CUBES

    CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]

    # Reset caches (important for each worker)
    RAIN_CUBES = {}
    SM_CUBES = {}

    FLOOD_CUBES = {}
    for depth in [10, 30]:
        area = load_3d_cube(
            f"{flood_dir}/{depth}cm/flooded_area_5km_total_Ens{ens_num}_23_{depth}cm.nc"
        )
        vol = load_3d_cube(
            f"{flood_dir}/{depth}cm/flooded_volume_5km_total_Ens{ens_num}_23_{depth}cm.nc"
        )

        FLOOD_CUBES[f"{depth}cm_area"] = prepare_flood_cube(area, CATCHMENT_POLY)
        FLOOD_CUBES[f"{depth}cm_volume"]  = prepare_flood_cube(vol, CATCHMENT_POLY)
         

def process_single_event_worker(args):
    (
        event_num,
        ncl_events,
        rainfall_cube_dir,
        sm_dir,
        ens_num,
        method,
        threshold_level,
    ) = args

    global FLOOD_CUBES, RAIN_CUBES, SM_CUBES, CATCHMENT_POLY

    details = get_rainfall_event_details(ncl_events, event_num)
    year = details['yr']

    # =====================================
    # 🌧️ Rainfall (cached per year)
    # =====================================
    if year not in RAIN_CUBES:
        rain_cube = get_rainfall_cube(year, ens_num, rainfall_cube_dir)
        rain_cube = filter_closer_to_catchment(rain_cube, CATCHMENT_POLY)
        RAIN_CUBES[year] = rain_cube
    else:
        rain_cube = RAIN_CUBES[year]

    mask_2d = build_mask(rain_cube, CATCHMENT_POLY, method=method)
    rain_cube_masked = apply_mask_to_cube(rain_cube, mask_2d)

    peak = find_max_precip_location(rain_cube_masked,mask_2d, details['start_idx'], details['stop_idx'])

    temp_profile_dict = find_temporal_profile( rain_cube_masked, details, peak, plot=False )

    # =====================================
    # 🌊 Flood (already cached)
    # =====================================
    flood_stats = {}

    for depth in [10, 30]:
        flood_area = FLOOD_CUBES[f"{depth}cm_area"]
        flood_area   = FLOOD_CUBES[f"{depth}cm_area"][event_num-1, :, :]
        # print(flood_area.shape)
        flood_vol  = FLOOD_CUBES[f"{depth}cm_volume"]
        flood_vol   = FLOOD_CUBES[f"{depth}cm_volume"][event_num-1, :, :]
        # print(flood_vol.shape)
        analysis = analyse_peak_event(
            rain_cube_masked,
            peak,
            threshold_level=threshold_level,
            flood_area_data=flood_area.data,
            flood_volume_data=flood_vol.data)

        for k, v in analysis.items():
            flood_stats[f"{depth}cm_{k}"] = v

    # =====================================
    # 🌱 Soil moisture (cached per year)
    # =====================================
    if year not in SM_CUBES:
        sm_file = f"{sm_dir}/r001i1p*****_{year-1}1201-{year}1130_mrso.nc"
        sm_cube = load_3d_cube(sm_file)
        sm_cube = filter_closer_to_catchment(sm_cube, CATCHMENT_POLY)
        SM_CUBES[year] = sm_cube
    else:
        sm_cube = SM_CUBES[year]

    sm_stats = find_sm_stats(sm_dir, year, CATCHMENT_POLY, rain_cube_masked, peak, details)
    # =====================================
    # Combine
    # =====================================
    return {**details, **peak, **temp_profile_dict, **flood_stats, **sm_stats}

def process_events_parallel_fast(event_nums, ncl_events, boundary_gdf,
                                rainfall_cube_dir, sm_dir, flood_dir,
                                ens_num, method, threshold_level,
                                n_workers=None):

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    args_list = [
        (
            event_num,
            ncl_events,
            rainfall_cube_dir,
            sm_dir,
            ens_num,
            method,
            threshold_level
        )
        for event_num in event_nums
    ]

    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(boundary_gdf, flood_dir, ens_num)
    ) as pool:

#         results = pool.map(process_single_event_worker, args_list)
        #results = list(tqdm(pool.imap(process_single_event_worker, args_list),total=len(args_list)))
        results = list(tqdm(pool.imap_unordered(process_single_event_worker, args_list),total=len(args_list),desc="Processing events"))

    return pd.DataFrame(results)


def process_events_serial_fast(event_nums, ncl_events, boundary_gdf,
                              rainfall_cube_dir, sm_dir, flood_dir,
                              ens_num, method, threshold_level):

    # Manually initialise (same as worker init)
    init_worker(boundary_gdf, flood_dir, ens_num)

    args_list = [
        (
            event_num,
            ncl_events,
            rainfall_cube_dir,
            sm_dir,
            ens_num,
            method,
            threshold_level
        )
        for event_num in event_nums
    ]

    results = []

    for args in tqdm(args_list, desc="Processing events (serial)"):
        results.append(process_single_event_worker(args))

    return pd.DataFrame(results)


# ENS_LIST = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
ENS_LIST = ['11', '12', '13', '15']


if __name__ == "__main__":

    all_results = []

    for ens in tqdm(ENS_LIST, desc="Ensembles"):

        tqdm.write(f"🚀 Running ensemble {ens}")

        # ✅ Define paths PER ensemble
        rainfall_cube_dir = f'/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens}/'
        flood_dir = f"/scratch/hydro5/users/la17355/FUTURE-FLOOD/Results/Pluvial/v4/5km_total/Ens{ens}_23/"
        sm_dir = f'/scratch/hydro4/shared_data/climate_projections/UKCP18/UKCP_local/Soil_moisture/5km_regridded/Ens_{ens}/'

        # ✅ Load events PER ensemble
        ncl_events = pd.read_csv(
            RAINFALL_CSV_DIR + f"Tyne(Northumberland)_{ens}_full_events_with_event_nums.csv"
        )

        event_nums_to_process = range(1, len(ncl_events))

        results = process_events_parallel_fast(
            event_nums        = event_nums_to_process,
            ncl_events        = ncl_events,
            boundary_gdf      = boundary_gdf,
            rainfall_cube_dir = rainfall_cube_dir,
            sm_dir            = sm_dir,
            flood_dir         = flood_dir,
            ens_num           = ens,
            method            = METHOD,
            threshold_level   = 0.3,
            n_workers         = 4
        )

        results["ensemble"] = ens

        # ✅ Save per ensemble (good practice)
        results.to_csv(f"results_ens_{ens}.csv", index=False)

        all_results.append(results)

    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("results_all_ensembles.csv", index=False)