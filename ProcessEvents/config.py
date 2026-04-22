import geopandas as gpd
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
MOLLY_DIR_FF     = "/scratch/hydro4/users/kv25483/FutureFlood/"
RAINFALL_CSV_DIR = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall_events/fixed_threshold_30mm_with_volume/"
OUT_DIR          = "/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/"
CATCHMENT_LOOKUP_FP = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/Data/CEH_catchments/CEH_IHU_with_coastline/hyd_areas_GB_with_subcatchments_no_spaces.csv"
ENSEMBLE_MEMBERS = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']

CATCHMENTS = gpd.read_file(MOLLY_DIR_FF + "Data/CatchmentShapefiles/hyd_areas_GB_with_subcatchments.shp")
catchment_lookup = pd.read_csv(CATCHMENT_LOOKUP_FP)
CATCHMENT_LOOKUP_DICT = dict(zip(catchment_lookup["HA_NUM"], catchment_lookup["HA_NAME"]))