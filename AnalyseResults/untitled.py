import os
import sys
import re
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import rioxarray as rxr
from tqdm import tqdm

from Functions_AnalyseResults import partial_r2_for_catchment, plot_with_best_fit_line

sys.path.append('../ProcessEvents/')
from config import CATCHMENT_LOOKUP_DICT, OUT_DIR # MOLLY_DIR_FF, RAINFALL_CSV_DIR, ENSEMBLE_MEMBERS, , CATCHMENTS

sys.path.append('../../ProcessCatchments')
from functions import *
from functions_stage2 import (get_rainfall_cube_subsection, setup_worker_logger, find_max_precip_location_new, 
                        find_temporal_profile_new, maybe_diagnose, analyse_peak_event,
                             plot_cluster_check)

from config import RAINFALL_CSV_DIR, CATCHMENT_LOOKUP_DICT, OUT_DIR, CATCHMENTS, ENSEMBLE_MEMBERS 
from Functions_AnalyseResults import partial_r2_for_catchment, plot_with_best_fit_line

all_catchments = set(CATCHMENT_LOOKUP_DICT.keys())

catchments_with_flood_output = []
for catchment_num in all_catchments:
    catchment_name = CATCHMENT_LOOKUP_DICT[catchment_num]
    fp = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/Catchment_{catchment_num}/{catchment_name}.pkl"
    flood_fp = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{catchment_num}/Ens01_{catchment_num}/10cm/flooded_area_5km_total_Ens01_{catchment_num}_10cm.nc"
        
    if os.path.isfile(flood_fp) and os.path.isfile(fp):
        catchments_with_flood_output.append(catchment_num)
print(len(catchments_with_flood_output))

rainfall_events_all = []
for catchment_num in catchments_with_flood_output:
    catchment_name = CATCHMENT_LOOKUP_DICT[catchment_num]
    fp = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/Catchment_{catchment_num}/{catchment_name}.pkl"
    rainfall_events = pd.read_pickle(fp)
    rainfall_events_complete = rainfall_events[rainfall_events['mismatch']!=True].copy()
    rainfall_events['catchment_num'] = catchment_num
    # print(f"Catchment {catchment_num} has {len(rainfall_events)}, of which {len(rainfall_events) - len(rainfall_events_complete)} are mismatched")
    rainfall_events_all.append(rainfall_events) 
rainfall_events_all_df = pd.concat(rainfall_events_all, ignore_index=True)    
rainfall_events_all_df.reset_index(inplace=True, drop = True)

# --- Output containers ---
peak_cell_slope_avg = []
peak_cell_slope_max = []
catchment_slope_avg = []
catchment_slope_max = []
peak_cell_sink_frac = []

# --- Caches ---
dem_cache = {}
slope_cache = {}
sink_cache = {}
catchment_clip_cache = {}

cellsize = 5000  # confirm this

# --- Loop ---
for row in tqdm(rainfall_events_all_df.itertuples(), total=len(rainfall_events_all_df)):

    catchment_num = row.catchment_num
    
    boundary_gdf = CATCHMENTS[CATCHMENTS['HA_NUM'] == str(catchment_num)] 
    CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]

    # =========================
    # LOAD / CACHE STATIC DATA
    # =========================
    if catchment_num not in dem_cache:

        dem_path = f"/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/Model_builds/Pluvial/v4/dem/dem_filled_30m_{catchment_num}.tif"
        sink_path = f"/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/Model_builds/Pluvial/v4/pluvial_sink_mask/pluvial_sink_mask_30m_{catchment_num}.tif"

        dem_da = rxr.open_rasterio(dem_path, masked=True).squeeze()
        sink_da = rxr.open_rasterio(sink_path, masked=True).squeeze()

        # --- Compute slope ONCE ---
        res_x, res_y = dem_da.rio.resolution()
        dz_dy, dz_dx = np.gradient(dem_da.values, abs(res_y), abs(res_x))
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        slope_da = dem_da.copy(data=slope_deg)

        # --- Cache everything ---
        dem_cache[catchment_num] = dem_da
        sink_cache[catchment_num] = sink_da
        slope_cache[catchment_num] = slope_da

        # --- Clip catchment ONCE ---
        slope_catchment = slope_da.rio.clip([CATCHMENT_POLY], drop=True)
        catchment_clip_cache[catchment_num] = slope_catchment

    # --- Retrieve from cache ---
    slope_da = slope_cache[catchment_num]
    sink_da = sink_cache[catchment_num]
    slope_catchment = catchment_clip_cache[catchment_num]

    # =========================
    # PEAK CELL CLIP
    # =========================
    yi = int(row.y_idx_global)
    xi = int(row.x_idx_global)

    try:
        year_cube = get_rainfall_cube_subsection(
            int(row.start_year),
            row.ens,
            f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{row.ens}/",
            int(row.start_idx) - 1,
            int(row.stop_idx))
    except ValueError as e:
        print(f"Skipping event {row.Index} (catchment {catchment_num}): {e}")
        peak_cell_slope_avg.append(np.nan)
        peak_cell_slope_max.append(np.nan)
        catchment_slope_avg.append(np.nan)
        catchment_slope_max.append(np.nan)
        peak_cell_sink_frac.append(np.nan)
        continue

    # Extract coordinates
    x_coord = year_cube.coord('projection_x_coordinate').points
    y_coord = year_cube.coord('projection_y_coordinate').points
    
    cx, cy = x_coord[xi], y_coord[yi]

    cell_xmin, cell_xmax = cx - cellsize/2, cx + cellsize/2
    cell_ymin, cell_ymax = cy - cellsize/2, cy + cellsize/2

    slope_cell = slope_da.rio.clip_box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)
    sink_cell = sink_da.rio.clip_box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)

    slope_vals = slope_cell.values
    sink_vals = sink_cell.values

    # =========================
    # SAFE STATS
    # =========================
    if np.all(np.isnan(slope_vals)):
        mean_slope = np.nan
        max_slope = np.nan
    else:
        mean_slope = np.nanmean(slope_vals)
        max_slope = np.nanmax(slope_vals)

    sink_frac = np.nan if np.all(np.isnan(sink_vals)) else np.nanmean(sink_vals > 0)

    # =========================
    # APPEND
    # =========================
    peak_cell_slope_avg.append(mean_slope)
    peak_cell_slope_max.append(max_slope)
    catchment_slope_avg.append(float(slope_catchment.mean()))
    catchment_slope_max.append(float(slope_catchment.max()))
    peak_cell_sink_frac.append(sink_frac)

# =========================
# FINAL SAFETY CHECK
# =========================
n = len(rainfall_events_all_df)
assert len(peak_cell_slope_avg) == n

# =========================
# ASSIGN
# =========================
rainfall_events_all_df["peak_cell_slope_avg"] = peak_cell_slope_avg
rainfall_events_all_df["peak_cell_slope_max"] = peak_cell_slope_max
rainfall_events_all_df["catchment_slope_avg"] = catchment_slope_avg
rainfall_events_all_df["catchment_slope_max"] = catchment_slope_max
rainfall_events_all_df["peak_cell_sink_frac"] = peak_cell_sink_frac

rainfall_events_all_df.to_csv("test.csv", index=False)