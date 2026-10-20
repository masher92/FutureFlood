import pandas as pd
import numpy as np

def prep_for_modelling (fp, remove_events_with_zero_flooding):
    
    ### Get event data
    rainfall_events_all_df = pd.read_pickle(fp)
    # Get rid of unphysical values
    rainfall_events_all_df = rainfall_events_all_df[rainfall_events_all_df['max_precip']<130].copy()
    # Add length variables, remove length '1' events
    rainfall_events_all_df["length"] = (rainfall_events_all_df["stop_idx"] - rainfall_events_all_df["start_idx"]).astype("float32")
    # rainfall_events_all_df = rainfall_events_all_df[rainfall_events_all_df['length_of_active_rain']>1].copy()
    # Add random variable
    rainfall_events_all_df['random_variable'] = np.random.normal(size=len(rainfall_events_all_df))

    # Set NAN values to 0 (as they should represent the same thing)
    rainfall_events_all_df[['peak_cell_outflow_N', 
                            'peak_cell_outflow_S', 'peak_cell_outflow_E','peak_cell_outflow_W']]=rainfall_events_all_df[[
        'peak_cell_outflow_N', 'peak_cell_outflow_S', 'peak_cell_outflow_E','peak_cell_outflow_W']].fillna(value=0)

    for level in ['t50', 't60', 't80']:
        rainfall_events_all_df[f'cluster_mean_intensity_{level}'] = (
            rainfall_events_all_df[f'cluster_rain_total_{level}'] / rainfall_events_all_df[f'cluster_n_cells_{level}'].replace(0, np.nan))    

    # Remove variables which carry no meaning
    variables = list(rainfall_events_all_df.columns)
    to_be_removed = {
                    #'x_coord', 'y_coord',
                    'x_idx', 'y_idx', 'x_idx_global', "y_idx_global", 't_global','t_local', "day_360", "month",
                     "max_precip_from_csv", "event_num", "time_mismatch", "ens",  'start_idx', 'stop_idx',
                     "t_local_rebased", "times", "values", "mismatch", 'n_valid', 'pct_over1', 'pct_under0',
                    # "10cm_volume", "30cm_area", "30cm_volume", "10cm_area", 
                    'mean_sm_3_before_event', 'mean_sm_4_before_event', 'mean_sm_5_before_event', 'neighbourhood_flood_10_area',
                     'neighbourhood_flood_10_vol', 'neighbourhood_flood_30_area', 'neighbourhood_flood_30_vol',
                    'threshold_flood_10_area_t50', 'cluster_flood_10_area_t50',  'threshold_flood_10_vol_t50',
                     'cluster_flood_10_vol_t50', 'threshold_flood_30_area_t50', 'cluster_flood_30_area_t50',
                     'threshold_flood_30_vol_t50','cluster_flood_30_vol_t50','threshold_flood_10_area_t60',
                    # 'cluster_flood_10_area_t60','cluster_flood_10_area_t80','cluster_flood_30_area_t80',
                    'threshold_flood_10_vol_t60','cluster_flood_10_vol_t60',
                     'threshold_flood_30_area_t60','cluster_flood_30_area_t60','threshold_flood_30_vol_t60',
                     'cluster_flood_30_vol_t60','threshold_flood_10_area_t80', 
                     'threshold_flood_10_vol_t80', 'cluster_flood_10_vol_t80', 'threshold_flood_30_area_t80',
                      'threshold_flood_30_vol_t80','cluster_flood_30_vol_t80', 
                     'event_num.1', 'date','Unnamed: 0', 'peak_cell_id',"rainfall_peak_day", "catchment_num",
                     'peak_cell_edge_truncated_N', 'peak_cell_edge_truncated_S', 'peak_cell_edge_truncated_E',
                     'peak_cell_edge_truncated_W',"peak_cell_near_catchment_boundary",
                     'total_acc_n'
                     #'se_min', 'se_max'
                    }

    rainfall_events_all_df = rainfall_events_all_df.drop(columns=to_be_removed, errors='ignore').copy()
    
    if remove_events_with_zero_flooding == True:
        rainfall_events_all_df=rainfall_events_all_df[rainfall_events_all_df['10cm_area']>0.1]
        
    return rainfall_events_all_df
    
def create_subset (df, sample_size, random_seed):
    
    ### Get smaller subset for fitting the model quicker (for testing)
    df= df.sample(n=sample_size, random_state=random_seed)  # adjust size to something that runs in seconds
    df.reset_index(inplace=True, drop=True)
    
    return df  

def create_temporal_subset (df, filter_before_or_after, filter_year):
    
    ### Get smaller subset for fitting the model quicker (for testing)
    if filter_before_or_after == 'before':
        df= df[df['start_year']<= filter_year]
    else:
        df= df[df['start_year']>= filter_year]
    df.reset_index(inplace=True, drop=True)
    
    return df     