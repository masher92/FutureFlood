import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from HelperFunctions import *  
from ProcessingFunctions import (load_flood_arrays, load_rainfall_events, add_hydro_year, find_wet_onset, sm_day_before,
                                  build_catchment_mask, load_soil_moisture,extract_at_cell, get_rainfall_data_from_climate_data,
                                  get_rainfall_at_peak_cells, find_peak_cell, add_rainfall_peak_time_vars, 
                                 find_spatial_stats_around_peak)

# ── Config ────────────────────────────────────────────────────────────────────
from config import RAINFALL_CSV_DIR, CATCHMENT_LOOKUP_DICT, OUT_DIR, CATCHMENTS, ENSEMBLE_MEMBERS 

FLOOD_ROOT = "/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/"
RAIN_ROOT = "/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails_v4/"
RAIN_CUBE_ROOT = f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/"
SM_ROOT =   '/scratch/hydro4/shared_data/climate_projections/UKCP18/UKCP_local/Soil_moisture/5km_regridded/'

WET_THRESH   = 0.1          # mm/hr; an hour counts as wet above this
FLOOD_DEPTHS = [10, 30]     # cm

full_rain_cube = get_rainfall_cube_subsection(2015, '01', f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/01/", 1, 2)

all_catchments = set(CATCHMENT_LOOKUP_DICT.keys())
catchments_with_flood_output = []
for catchment_num in all_catchments:
    catchment_name = CATCHMENT_LOOKUP_DICT[catchment_num]
    fp = RAIN_ROOT + f"Catchment_{catchment_num}/{catchment_name}.pkl"
    flood_fp = FLOOD_ROOT + f"Catchment_{catchment_num}/Ens01_{catchment_num}/10cm/flooded_area_5km_total_Ens01_{catchment_num}_10cm.nc"
        
    if os.path.isfile(flood_fp):
        catchments_with_flood_output.append(catchment_num)    
        
        
for catchment_num in catchments_with_flood_output[60:]: # catchments_with_flood_output:
     
    # Make directory for outputs, if it doesn't already exist
    print(f"Running for {catchment_num}")
    if not os.path.exists(os.path.join(OUT_DIR, f"Catchment_{catchment_num}")):
        os.mkdir(os.path.join(OUT_DIR, f"Catchment_{catchment_num}"))

    # If output already exists, then skip this catchment
    catchment_name  = CATCHMENT_LOOKUP_DICT[str(catchment_num)]  
    catchment_fp = os.path.join(OUT_DIR, f"Catchment_{catchment_num}", f"{catchment_name}_new_Nov.pkl")
    print(catchment_fp)
    if os.path.exists(catchment_fp):
        print("Already exists")
        continue        
        
    # Get supporting spatial data and create mask 
    boundary_gdf    = CATCHMENTS[CATCHMENTS['HA_NUM'] == str(catchment_num)]
    CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]         
    mask, mask_sub = build_catchment_mask(full_rain_cube, CATCHMENT_POLY)

    # Create structure to store results for whole catchment
    this_catchment_all_results = []
    
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────
    #    Loop through each ensemble member for this catchment
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────    
    
    for ens_num in tqdm(ENSEMBLE_MEMBERS):
        print(f"Running for EM {ens_num}")
        member_fp = os.path.join(OUT_DIR,f"Catchment_{catchment_num}",f"{catchment_name}_EM{ens_num}_new_Nov.pkl")
        
        # If output already exists, then skip this ensemble member
        if os.path.exists(member_fp):
            this_catchment_all_results.append(pd.read_pickle(member_fp)); continue

        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        # Get rainfall event data, add variables AND get flood data
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        events = add_hydro_year(load_rainfall_events(RAINFALL_CSV_DIR, catchment_name, ens_num))
        flood_cubes, flood_numpy = load_flood_arrays(FLOOD_ROOT, catchment_num, ens_num, mask, CATCHMENT_POLY)   
        
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        # Create structure to store results
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        this_ens_all_results = []
        
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        # Loop through each year and all the events in that year
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────        
        for hydro_year, events_in_year in tqdm(events.groupby('hydro_year')):
            
            # ─────────────────────────────────────────────────────────────────────────────────────────────────────
            # Get soil moisture data for this year  
            # ─────────────────────────────────────────────────────────────────────────────────────────────────────
            sm = load_soil_moisture(SM_ROOT, events_in_year, ens_num, hydro_year, CATCHMENT_POLY, mask)

            # ─────────────────────────────────────────────────────────────────────────────────────────────────────
            # Loop through all events within this 'hydro-year'
            # ─────────────────────────────────────────────────────────────────────────────────────────────────────
            
            for row in events_in_year.itertuples():
                
                # Get event number
                event_idx = int(row.event_num) - 1
                
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                #                  Create results dictionary for this event with basic details 
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                
                result = {'ens': ens_num, 'hydro_year': hydro_year, 'event_num': int(row.event_num),
                        'start_idx': int(row.start_indices), 'stop_idx': int(row.stop_indices),
                        'max_precip_from_csv': float(row.peaks),
                        'start_year': int(row.start_year), 'start_month': int(row.start_month),
                        'start_day': int(row.start_day),   'start_hour': int(row.start_hour),
                        'event_durations': int(row.event_durations)}

                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                #                Get rainfall data for the hydro-year this event happened in 
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────

                rainfall_year_cube, time_coord, time_mismatch = get_rainfall_data_from_climate_data(RAIN_CUBE_ROOT, hydro_year,
                                                                                                    ens_num, row, mask, CATCHMENT_POLY)
                result['time_mismatch'] = time_mismatch

                # Subset to catchment and get spatial offsets (so that cell references for within the catchment can also
                # be used on data that hasn't been subset to the catchment)
                rainfall_year_cube, x_offset, y_offset = subset_cube_to_bbox(rainfall_year_cube, CATCHMENT_POLY, buffer=0)

                # Check flood, rainfall and soil moisture data are all the same size
                # assert flood_numpy[10]['vol'].shape[1:] == rainfall_year_cube.shape[1:]  == sm.data.shape[1:]

                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                #           Find the cell with the peak value in both rainfall and flooding data 
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────   
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────

                rain_pk  = find_peak_cell(rainfall_year_cube.data)  # (t, y, x)
                flood_pk = find_peak_cell(flood_numpy[10]['area'][event_idx])     # (y, x)

                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                #   Extract the rainfall data at both of those locations
                #       - Record whether the two peaks were recorded at same location, and how far apart they are
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────   
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────

                result = get_rainfall_at_peak_cells (rain_pk, "rainpeak", rainfall_year_cube, flood_numpy, sm,  x_offset, y_offset, 
                                                     event_idx, WET_THRESH, result)
                result = get_rainfall_at_peak_cells (flood_pk, "floodpeak", rainfall_year_cube, flood_numpy, sm,  x_offset, y_offset,
                                                     event_idx, WET_THRESH, result)

                if rain_pk and flood_pk:
                    result['peaks_same_cell'] = (rain_pk['y_idx'], rain_pk['x_idx']) == (flood_pk['y_idx'], flood_pk['x_idx'])
                    # Find the distance (using the hypotenuse) between the two peak cells
                    result['peaks_dist_cells'] = float(np.hypot(rain_pk['y_idx'] - flood_pk['y_idx'],
                                                                rain_pk['x_idx'] - flood_pk['x_idx']))    
                else:
                    result['peaks_same_cell'] =np.nan
                    result['peaks_dist_cells'] = np.nan

                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                #   Record the time at which the rainfall peak occurred in the rainfall data
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────   
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────            
                result = add_rainfall_peak_time_vars(time_coord, result)

                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                #   Record the time at which the rainfall peak occurred in the rainfall data
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────   
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────   
                result = find_spatial_stats_around_peak(result, rainfall_year_cube, flood_numpy, flood_cubes, 
                                         CATCHMENT_POLY,boundary_gdf, 1, [0.5, 0.6, 0.8], plot=False)
                
                result = pd.DataFrame([result])
                
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────
                #   Add to ensemble member results, and catchment results
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────   
                # ─────────────────────────────────────────────────────────────────────────────────────────────────────  
                # print(row.peaks, result['rainpeak_max_precip'])

                this_ens_all_results.append(result)
                this_catchment_all_results.append(result)

        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────
        #   Save results for this ensemble member
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────   
        # ─────────────────────────────────────────────────────────────────────────────────────────────────────                 
        this_ens_all_results_df = pd.concat(this_ens_all_results, ignore_index=True).to_pickle(member_fp)
        
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────
    #   Save results for this catchment
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────   
    # ─────────────────────────────────────────────────────────────────────────────────────────────────────    
    this_catchment_all_results_df = pd.concat(this_catchment_all_results, ignore_index=True) 
    this_catchment_all_results_df.to_pickle(catchment_fp)    

    # Delete individual ensemble member results
    for ens_num in ENSEMBLE_MEMBERS:
        member_fp = os.path.join(OUT_DIR,f"Catchment_{catchment_num}",f"{catchment_name}_EM{ens_num}_new_Nov.pkl")
        if os.path.exists(member_fp):
            os.remove(member_fp)