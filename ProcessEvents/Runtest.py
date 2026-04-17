import numpy as np
import pandas as pd
import cftime
import geopandas as gpd
import re
import time
import gc
import os

from functions_clean import * 

def get_data_at_peak_cell(cube, one_event_results):

    x_idx = one_event_results['x_idx']
    y_idx = one_event_results['y_idx']

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
    cube_at_peak = cube[:, x_idx, y_idx]
    return cube_at_peak  

# # ── Config ────────────────────────────────────────────────────────────────────
MOLLY_DIR_FF     = "/scratch/hydro4/users/kv25483/FutureFlood/"
RAINFALL_CSV_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall_events/fixed_threshold_30mm_with_volume/"
OUT_DIR          = "/scratch/hydro4/users/kv25483/FutureFlood/Data/PeakIndices/"

catchments       = gpd.read_file(MOLLY_DIR_FF + "Data/CatchmentShapefiles/hyd_areas_GB_with_subcatchments.shp")
catchment_lookup = pd.read_csv("/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/CEH_catchments/CEH_IHU_with_coastline/hyd_areas_GB_with_subcatchments_no_spaces.csv")
catchment_lookup_dict = dict(zip(catchment_lookup["HA_NUM"], catchment_lookup["HA_NAME"]))

files             = os.listdir("/scratch/hydro5/users/la17355/FUTURE-FLOOD/Results/Pluvial/v4/tifs/")
catchment_numbers = {re.search(r'Ens15_(.+)', f).group(1) for f in files if re.search(r'Ens15_(.+)', f)}
catchments_to_run = sorted(catchment_numbers)
print(f"{len(catchments_to_run)} catchments to process")

for catchment_num in catchments_to_run:
    if catchment_num != '105':
        catchment_name   = catchment_lookup_dict[str(catchment_num)]
        print(f"Running for {catchment_name}")
        catchment_out    = os.path.join(OUT_DIR, f"Catchment_{catchment_num}")
        out_path         = os.path.join(catchment_out, "peak_indices.csv")
        os.makedirs(catchment_out, exist_ok=True)

        #### Prepare data for this catchment and ensemble member
        boundary_gdf    = catchments[catchments['HA_NUM'] == str(catchment_num)]
        _CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]

        ENSEMBLE_MEMBERS = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']

        results = []

        for ens_num in ENSEMBLE_MEMBERS:

            gc.collect()

            rainfall_events = pd.read_csv(
                RAINFALL_CSV_DIR + f"{catchment_name}_{ens_num}_full_events_with_event_nums.csv")

            rainfall_cube_dir = f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens_num}/"

            # Add event numbers
            rainfall_events['event_num'] = range(1, len(rainfall_events) + 1)

            print(f"Running for {catchment_name}, for {len(rainfall_events)} events for EM: {ens_num}")

            # ---- CACHE EVENT DETAILS (avoid recomputation) ----
            event_details_cache = {
                ev: get_rainfall_event_details(rainfall_events, ev)
                for ev in rainfall_events['event_num']}

            # Add year column
            rainfall_events['year'] = rainfall_events['event_num'].map(lambda ev: event_details_cache[ev]['yr'])

            # Create a mask applicable to all years
            start_time1 = time.time()
            full_rain_cube = get_rainfall_cube(2015, ens_num, rainfall_cube_dir)   # any year, grid is same
            FULL_MASK_2D = mask_cube_with_catchment_full_grid(full_rain_cube[0], _CATCHMENT_POLY, method='full_cell')
            print(f"Creating a mask for EM{ens_num} in {round(time.time() - start_time1, 2)}s")    

            # ---- PROCESS BY YEAR ----
            for year, events_in_year in rainfall_events.groupby('year'):
                start_time_total = time.time()
                print(f"\nProcessing year {year} with {len(events_in_year)} events")

                # ---- LOAD CUBE and subset to catchment boundaries ----
                start_time1 = time.time()
                year_cube = get_rainfall_cube(year, ens_num, rainfall_cube_dir)
                year_cube, x_offset, y_offset = subset_cube_to_bbox(year_cube, _CATCHMENT_POLY, buffer=0)
                print(f"Loaded cube for {year} in {round(time.time() - start_time1, 2)}s")

                # ---- Create version of mask for this year and trimmed to catchment
                nx_sub = year_cube.shape[2]
                ny_sub = year_cube.shape[1]
                mask_2d_sub = FULL_MASK_2D[y_offset:y_offset+ny_sub, x_offset:x_offset+nx_sub]

                # Cache time coordinate once
                time_coord = year_cube.coord('time')

                # ---- PROCESS EVENTS ----
                for row in events_in_year.itertuples():

                    event_num = int(row.event_num)
                    event_details = event_details_cache[event_num]
                    print(event_details['max_precip_from_csv'])
                    print(np.nanmax(year_cube[event_details['start_idx']:event_details['stop_idx'],:,:].data))

                    #print(f"Ens: {ens_num}, event num: {event_num}, event year: {year}")
                    print(f"Event num: {event_num}")

                    this_event_results = find_max_precip_location(year_cube, event_details['start_idx'], event_details['stop_idx'],
                        x_offset = x_offset, y_offset = y_offset, mask_2d=mask_2d_sub)

                    # Metadata
                    this_event_results['ens'] = ens_num
                    this_event_results['year'] = year
                    this_event_results['start_idx'] = event_details['start_idx']
                    this_event_results['stop_idx'] = event_details['stop_idx']

                    # Peak day
                    t = time_coord.units.num2date(
                        time_coord.points[this_event_results['t_global']])
                    this_event_results['rainfall_peak_day'] = cftime.Datetime360Day(t.year, t.month, t.day)

                    ### 
                    rainfall_at_peak = get_data_at_peak_cell(year_cube, this_event_results)
                    temp_profile_dict = find_temporal_profile(rainfall_at_peak, this_event_results, True)
                    this_event_results = {**this_event_results, **temp_profile_dict}

                    results.append(this_event_results)

                print(f"Completed operation for EM {ens_num}, {year} in {round(time.time() - start_time_total, 2)}s")

                # ---- CLEAN UP MEMORY ----
                del year_cube
                gc.collect()

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{catchment_name}.csv", index=False)      