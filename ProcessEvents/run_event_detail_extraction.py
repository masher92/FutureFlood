import numpy as np
import pandas as pd
import cftime
import geopandas as gpd
import re
import time
import gc
import os
import shapely

from functions_clean import * 

# ── Config ────────────────────────────────────────────────────────────────────
MOLLY_DIR_FF     = "/scratch/hydro4/users/kv25483/FutureFlood/"
RAINFALL_CSV_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall_events/fixed_threshold_30mm_with_volume/"
OUT_DIR          = "/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/"

# ── Get a list of catchments to run ───────────────────────────────────────────
catchments       = gpd.read_file(MOLLY_DIR_FF + "Data/CatchmentShapefiles/hyd_areas_GB_with_subcatchments.shp")
catchment_lookup = pd.read_csv("/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/CEH_catchments/CEH_IHU_with_coastline/hyd_areas_GB_with_subcatchments_no_spaces.csv")
catchment_lookup_dict = dict(zip(catchment_lookup["HA_NUM"], catchment_lookup["HA_NAME"]))

all_catchments = set(catchment_lookup_dict.keys())
files = os.listdir("/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/")
completed_catchments = {re.search(r'Catchment_(.+)', f).group(1) for f in files if re.search(r'Catchment_(.+)', f)}
catchments_to_run = all_catchments - completed_catchments
catchments_to_run = (all_catchments - completed_catchments) - {105}


# ── Loop over catchments, running main script ───────────────────────────────────

for catchment_num in catchments_to_run:
    # Define name of catchment and set-up an output directory
    catchment_name   = catchment_lookup_dict[str(catchment_num)]
    catchment_out    = os.path.join(OUT_DIR, f"Catchment_{catchment_num}")
    os.makedirs(catchment_out, exist_ok=True)

    # Prepare spatial data for this catchment and ensemble member
    boundary_gdf    = catchments[catchments['HA_NUM'] == str(catchment_num)]
    _CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]

    # Create a list to store ALL results (all ensemble members)
    results = []
    
    # ── Loop over ensemble members, running main script ──────────────────────────
    ENSEMBLE_MEMBERS = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
    for ens_num in ENSEMBLE_MEMBERS:
        # Gets rid of things that no longer need to be stored
        gc.collect()
        
        # Create a list to store results  for this ensemble member
        results_this_ens = []
        
        # Get a dataframe containing details of extreme rainfall events
        rainfall_events = pd.read_csv(RAINFALL_CSV_DIR + f"{catchment_name}_{ens_num}_full_events_with_event_nums.csv")
        # Add variable giving each event a number
        rainfall_events['event_num'] = range(1, len(rainfall_events) + 1)
        # Add year column
        rainfall_events['year'] = rainfall_events['event_num'].map(lambda ev: event_details_cache[ev]['yr'])
        
        # Establish filepath where netCDF of rainfall data lives for this ensemble member
        rainfall_cube_dir = f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens_num}/"
        
        # Print initialisation statement (now we know the number of events) 
        print(f"Running for {catchment_name}, for {len(rainfall_events)} events for EM: {ens_num}")

        # ---- CACHE EVENT DETAILS (avoid recomputation) ----
        # Create a dictionary that stores the output of pre-computing and storing the output of get_rainfall_event_details() 
        # for every event, keyed by event number. 
        event_details_cache = {
            ev: get_rainfall_event_details(rainfall_events, ev)
            for ev in rainfall_events['event_num']}

        # Create a mask which masks out any cells not within the catchment
        # This is year agnostic, and is applicable to all years
        # Has shape 244, 180 (full GB)
        # Create mask for whole country, and then trim it to the same extent as the rainfall cube before applying it
        # This allows creating mask just once, and then applying for each year
        start_time1 = time.time()
        full_rain_cube = get_rainfall_cube(2015, ens_num, rainfall_cube_dir)   # any year, grid is same
        FULL_MASK_2D = mask_cube_with_catchment_full_grid(full_rain_cube[0], _CATCHMENT_POLY, method='full_cell')
        print(f"Creating a mask for EM{ens_num} in {round(time.time() - start_time1, 2)}s")    

        # ---- PROCESS BY YEAR ----
        for year, events_in_year in rainfall_events.groupby('year'):
            start_time_total = time.time()
            print(f"\nProcessing year {year} with {len(events_in_year)} events")

            # ---- LOAD CUBE for this year and subset to catchment boundaries ----
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

            # ---- PROCESS for each EVENT in this year ----
            for row in events_in_year.itertuples():
                
                # Get event details for this event
                event_num = int(row.event_num)
                event_details = event_details_cache[event_num]
                #  print(f"max precip from csv: {event_details['max_precip_from_csv']}")
                # print(f"max precip from cube: {np.nanmax(year_cube[event_details['start_idx']:event_details['stop_idx'],:,:].data)}")
                #print(f"Ens: {ens_num}, event num: {event_num}, event year: {year}")
                print(f"Event num: {event_num}")
                
                # Returns the location in time and space, and the value, of the maximum precipitation value
                # x_idx, y_idx records its location in the subsetted cube (subsetted to the catchment boundaries)
                # x_idx_global, y_idx_global records its location in the whole UK cube
                this_event_results = find_max_precip_location(year_cube, event_details['start_idx'], event_details['stop_idx'],
                    x_offset = x_offset, y_offset = y_offset, mask_2d=mask_2d_sub)

                # Metadata
                this_event_results['ens'] = ens_num
                this_event_results['year'] = event_details['start_year']
                this_event_results['start_idx'] = event_details['start_idx']
                this_event_results['stop_idx'] = event_details['stop_idx']

                # Peak day
                t = time_coord.units.num2date(
                    time_coord.points[this_event_results['t_global']])
                this_event_results['rainfall_peak_day'] = cftime.Datetime360Day(t.year, t.month, t.day)

                # Rainall_at_peak is a 1D cube, containing the rainfall data for a full year at the grid cell where the maximum occurred
                # The find_temporal_profile function then searches this 1D cube and extracts rainfall between the start and stop index
                # These rainfall values and times are saved and then also a number of temporal profile variables are calculated
                # These are added to the overall results dictionary
                rainfall_at_peak = get_data_at_peak_cell(year_cube, this_event_results)
                temp_profile_dict = find_temporal_profile(rainfall_at_peak, this_event_results, plot=False)
                this_event_results = {**this_event_results, **temp_profile_dict}
                
                # Add to the overall list of results and the results for one EM
                results.append(this_event_results)
                results_this_ens.append(this_event_results)

            print(f"Completed operation for EM {ens_num}, {year} in {round(time.time() - start_time_total, 2)}s")
            
            # Save the results for this EM
            results_this_ens_df = pd.DataFrame(results_this_ens)
            results_this_ens_df.to_pickle(os.path.join(catchment_out, f"{catchment_name}_EM{ens_num}.pkl"))
            
            # ---- CLEAN UP MEMORY ----
            del year_cube
            gc.collect()
    
    # Covnert overall results to dataframe and save to a pickle file
    results_df = pd.DataFrame(results)
    results_df.to_pickle(os.path.join(catchment_out, f"{catchment_name}.pkl"))
    
    # Clean up per-ensemble files
    if os.path.exists(os.path.join(catchment_out, f"{catchment_name}.pkl")):
        print("Produced overall output")
        for ens_num in ENSEMBLE_MEMBERS:
            output_fp = os.path.join(catchment_out, f"{catchment_name}_EM{ens_num}.pkl")
            os.remove(output_fp)
            print(f"Deleted {output_fp}")
    else:
        print("WARNING: Combined file not found — individual ensemble files kept as fallback.")