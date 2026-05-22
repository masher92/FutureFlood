import numpy as np
import pandas as pd
import re
import time
import gc
import os
import sys
import math
import glob
import datetime
import geopandas as gpd
from shapely.geometry import MultiPolygon
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import Counter

from functions import *
from functions_stage2 import (get_rainfall_cube_subsection,get_rainfall_cube_subsection_notbc, setup_worker_logger, 
                              find_max_precip_location_new, find_temporal_profile_new, maybe_diagnose, analyse_peak_event,
                             plot_cluster_check, plot_n_peaks, get_exceedance_summary, plot_surrounding_ts, check_extremes_across_combos, 
                              plot_surrounding_ts_new, find_problematic_catchment_em_combos, find_problematic_catchment_em_combos_complex)

# ── Config ────────────────────────────────────────────────────────────────────
from config import RAINFALL_CSV_DIR, CATCHMENT_LOOKUP_DICT, OUT_DIR, CATCHMENTS, ENSEMBLE_MEMBERS 

# ── Define which catchments to run ────────────────────────────────────────────────────────────────────
all_catchments = set(CATCHMENT_LOOKUP_DICT.keys())

print("running finding problematic catchemnts")
results_df, problematic_catchment_names, problematic_catchment_nums = find_problematic_catchment_em_combos_complex(all_catchments, CATCHMENT_LOOKUP_DICT, ENSEMBLE_MEMBERS, RAINFALL_CSV_DIR)
results_df.sort_values(by='% Problematic', ascending = False)[:5]

print("running check extremes")
results = check_extremes_across_combos(results_df, plot=False)
results.to_csv("results.csv", index=False)