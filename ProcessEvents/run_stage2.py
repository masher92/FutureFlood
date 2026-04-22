import pandas as pd
import numpy as np
import geopandas as gpd
import os
import re
import iris
import cftime 
import gc
import glob
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
import time 

from functions import get_rainfall_cube, load_3d_cube, prepare_flood_cube, mask_cube_with_catchment_full_grid
from functions_stage2 import run_spatial_diagnostics, maybe_diagnose, get_data_at_peak_cell, plot_peak_check
from config import MOLLY_DIR_FF, RAINFALL_CSV_DIR, OUT_DIR, ENSEMBLE_MEMBERS, CATCHMENT_LOOKUP_DICT, CATCHMENTS

# Get list of catchments to run
all_catchments = set(CATCHMENT_LOOKUP_DICT.keys())
files = os.listdir(OUT_DIR)
completed_catchments = {re.search(r'Catchment_(.+)', f).group(1) for f in files if re.search(r'Catchment_(.+)', f)}
catchments_to_run = (all_catchments - completed_catchments) - {105} - {14}

def stage2_procesing(catchment_num):
    catchment_name = CATCHMENT_LOOKUP_DICT[str(catchment_num)]
    boundary_gdf   = CATCHMENTS[CATCHMENTS['HA_NUM'] == str(catchment_num)]
    CATCHMENT_POLY = boundary_gdf.geometry.iloc[0]

    # ── Load event details for this catchment ─────────────────────────────────────
    # Each row is one rainfall event: contains peak timing, spatial indices, ensemble
    # member, year, and pre-computed indices into the rainfall/flood cubes
    rainfall_events = pd.read_pickle(f"../../Data/EventDetails/Catchment_{catchment_num}/{catchment_name}.pkl")

    # ── Datetime conversions ──────────────────────────────────────────────────────
    # Convert peak day string to pandas Timestamp for arithmetic and .dt accessor
    rainfall_events['rainfall_peak_day'] = pd.to_datetime(
        rainfall_events['rainfall_peak_day'], format="%Y-%m-%d %H:%M:%S")

    # Also store as cftime.Datetime360Day so we can directly compare against iris
    # time coordinates, which use the 360-day calendar internally. Without this
    # conversion we would get type errors when doing datetime arithmetic against
    # the SM cube's time axis.
    rainfall_events['rainfall_peak_360'] = rainfall_events['rainfall_peak_day'].apply(
        lambda d: cftime.Datetime360Day(d.year, d.month, d.day, d.hour, d.minute, d.second))

    # ── 360-day calendar day indices ──────────────────────────────────────────────
    # hydro_day_360: day-of-hydrological-year on the 360-day calendar.
    #   The hydrological year runs Dec–Nov, so December is month 0 (12 % 12 = 0),
    #   giving day indices 0–359 starting from 1 Dec.
    #   Used to slice the SM cube, which is indexed on the hydrological year.
    rainfall_events['hydro_day_360'] = (
        (rainfall_events['rainfall_peak_day'].dt.month % 12) * 30
        + rainfall_events['rainfall_peak_day'].dt.day)

    # day_360: day-of-calendar-year on the 360-day calendar (Jan = month 0).
    #   Kept for reference / other uses but not used in the main loop below.
    rainfall_events['day_360'] = (
        (rainfall_events['rainfall_peak_day'].dt.month - 1) * 30
        + rainfall_events['rainfall_peak_day'].dt.day)

    # t_local: index of the peak timestep within the rainfall cube slice for this
    #   event. t_global is the absolute index across the full hourly rainfall cube;
    #   start_idx is where this event's window begins, so t_local = t_global - start_idx
    #   gives the position of the peak within the [start_idx:stop_idx] slice.
    rainfall_events['t_local'] = rainfall_events['t_global'] - rainfall_events['start_idx']

    # event_num: global event counter across all ensembles and years for this
    #   catchment. Used for diagnostic labelling only.
    rainfall_events['event_num'] = range(1, len(rainfall_events) + 1)

    # event_num_this_ens: event counter that resets to 1 at the start of each
    #   ensemble member. This is the correct index into the flood cubes' first
    #   dimension, because each ensemble's flood cube only contains that ensemble's
    #   events (not events from other ensemble members).
    rainfall_events['event_num_this_ens'] = rainfall_events.groupby('ens').cumcount() + 1

    # ── Hydraulic conductivity — load and realise once per catchment ──────────────
    # HC is a static field (no time dimension): one value per grid cell.
    # We load the iris cube (needed by the diagnostic plotter which calls iplt),
    # then immediately realise the underlying numpy array so the row loop can use
    # fast numpy indexing rather than repeated iris constraint evaluation.
    # Shape: (y, x)
    HC_CUBE = iris.load(f"/scratch/hydro4/users/kv25483/FutureFlood/Data/HydraulicConductivity/5km_{catchment_num}.nc")[0]
    HC_DATA = HC_CUBE.data  # realised numpy array — reused every row, zero iris overhead

    flood_dir = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{catchment_num}/"

    n_days_ls      = [2, 3, 4, 5]   # antecedent SM windows to compute (days before peak)
    all_results    = []              # accumulates one dict per event; converted to DataFrame at end
    DIAGNOSE_EVENT = [1, 2, 3, 4, 5]  # event_num values for which to produce diagnostic plots

    # flood_cubes: iris CubeList — kept so the diagnostic plotter (which calls
    #   iplt.pcolormesh) receives proper iris objects with coordinate metadata.
    # flood_numpy: plain numpy arrays extracted from the same cubes — used in the
    #   row loop for fast indexing without iris overhead.
    # Both are keyed [depth]['area' | 'vol'].
    flood_cubes = {}
    flood_numpy = {}
    current_ens = None  # sentinel: tracks which ensemble member's flood cubes are loaded

    for (ens_num, year), group in rainfall_events.groupby(['ens', 'year']):

        rainfall_cube_dir = f"/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/{ens_num}/"
        print(f"Processing ens={ens_num}, year={year} ({len(group)} events)")

        # ── Reload flood cubes when the ensemble member changes ───────────────────
        # Flood cubes contain one time step per event for a given ensemble member,
        # covering ALL years of that ensemble. They are therefore constant across
        # years and only need reloading when we move to a new ensemble member.
        # Deleting the old cubes before loading the new ones keeps peak memory low.
        if ens_num != current_ens:
            if flood_cubes:
                del flood_cubes, flood_numpy
                gc.collect()

            ens_flood_dir = f"{flood_dir}/Ens{ens_num}_{catchment_num}/"
            flood_cubes = {}
            flood_numpy = {}

            for depth in [10, 30]:
                # Load and prepare the iris cube (prepare_flood_cube applies any
                # necessary coordinate fixes / regridding defined in functions.py)
                area_cube = prepare_flood_cube(load_3d_cube(
                    f"{ens_flood_dir}/{depth}cm/flooded_area_5km_total_Ens{ens_num}_{catchment_num}_{depth}cm.nc"))
                vol_cube  = prepare_flood_cube(load_3d_cube(
                    f"{ens_flood_dir}/{depth}cm/flooded_volume_5km_total_Ens{ens_num}_{catchment_num}_{depth}cm.nc"))

                # Keep iris cubes for the diagnostic plotter
                flood_cubes[depth] = {'area': area_cube, 'vol': vol_cube}

                # Realise numpy arrays now so the row loop never touches iris again.
                # Shape: (n_events_this_ens, y, x)
                flood_numpy[depth] = {'area': area_cube.data, 'vol': vol_cube.data}

            current_ens = ens_num

        # ── Load soil moisture cube and realise to numpy once per (ens, year) ─────
        # Each SM file covers one hydrological year: 1 Dec (year-1) to 30 Nov (year).
        # We load the iris cube so the diagnostic plotter can use it, then immediately
        # pull the full data array into memory. Realising upfront means every
        # subsequent row-level access is a plain numpy slice — no repeated iris I/O.
        # Shape after realisation: (360, y, x) — one time step per day of the 360-day year.
        sm_dir  = f'/scratch/hydro4/shared_data/climate_projections/UKCP18/UKCP_local/Soil_moisture/5km_regridded/Ens_{ens_num}/'
        sm_file = glob.glob(f"{sm_dir}/r001i1p*****_{year-1}1201-{year}1130_mrso.nc")[0]
        sm      = iris.load(sm_file)[0]
        sm_data = sm.data  # realise: shape (360, y, x)

        # ── Build SM time axis once per (ens, year) group ────────────────────────
        # Extract the time coordinate points as a numpy array of cftime objects.
        # We use any single spatial cell (here: the peak cell of the first event in
        # this group) just to get the 1-D time axis — the time coordinate is the
        # same regardless of which cell we extract. Doing this outside the row loop
        # avoids rebuilding the array for every event.
        sm_example  = get_data_at_peak_cell(sm, group.iloc[0], 'x_idx_global', 'y_idx_global')
        sm_time_arr = np.array([cell.point for cell in sm_example.coord('time').cells()])

        start_time = time.time()

        for row in group.itertuples():

            # ── Diagnostic plots (uses iris cubes for proper coordinate-aware plotting)
            # maybe_diagnose is a no-op unless row.event_num is in DIAGNOSE_EVENT,
            # so there is no performance cost for non-diagnostic rows.
            # The iris cubes are passed here because plot_peak_check calls
            # iplt.pcolormesh, which requires an iris cube with spatial coordinates.
            maybe_diagnose(
                row,
                event_num=row.event_num,
                condition=(row.event_num in DIAGNOSE_EVENT),
                sm=sm,
                HC_CUBE=HC_CUBE,
                flood_cubes=flood_cubes,
                catchment_poly=CATCHMENT_POLY,
                ens_num=ens_num,
                rainfall_cube_dir=rainfall_cube_dir)

            # ── Spatial indices for this event's peak rainfall cell ───────────────
            # x_idx_global / y_idx_global are integer grid indices into the 5 km
            # national grid. Using them directly avoids any coordinate-based lookup.
            xi = int(row.x_idx_global)
            yi = int(row.y_idx_global)

            # ── Antecedent soil moisture stats ────────────────────────────────────
            # Extract the full time series at the peak cell as a 1-D numpy array
            # (length 360), then apply a boolean mask to select the n_days window
            # immediately before the event peak. This is much faster than building
            # an iris Constraint inside the loop because all operations are numpy.
            sm_at_peak_data   = sm_data[:, yi, xi]  # shape: (360,)
            rainfall_peak_360 = row.rainfall_peak_360

            mean_sm_stats = {}
            for n_days in n_days_ls:
                delta = datetime.timedelta(days=n_days)

                # Select time steps in the half-open window [peak - n_days, peak).
                # Both sm_time_arr and rainfall_peak_360 are cftime.Datetime360Day
                # objects, so subtraction and comparison are valid.
                mask        = (sm_time_arr >= (rainfall_peak_360 - delta)) & (sm_time_arr < rainfall_peak_360)
                subset_data = sm_at_peak_data[mask]

                if subset_data.size == 0:
                    # This can happen for events near the very start of the
                    # hydrological year (early December), where the n_days window
                    # would reach back into the previous file which is not loaded.
                    print(f"  Warning: empty SM window for event={row.event_num}, "
                          f"n_days={n_days}, peak={rainfall_peak_360}")
                    mean_sm_stats[f'mean_sm_{n_days}_before_event'] = float('nan')
                    continue

                mean_sm_stats[f'mean_sm_{n_days}_before_event'] = float(subset_data.mean())

            # ── Flood stats ───────────────────────────────────────────────────────
            # Index into the pre-realised numpy flood arrays.
            # Dimension 0 is event number within this ensemble member (0-based),
            # so we use event_num_this_ens - 1. Using event_num - 1 (the global
            # counter) would be wrong because it would index into a different
            # ensemble member's events once we move past the first ensemble.
            # Dimensions 1 and 2 are the spatial y and x grid indices.
            flood_stats = {}
            for depth in [10, 30]:
                flood_stats[f"{depth}cm_area"]   = flood_numpy[depth]['area'][row.event_num_this_ens - 1, yi, xi]
                flood_stats[f"{depth}cm_volume"] = flood_numpy[depth]['vol'][row.event_num_this_ens - 1, yi, xi]

            # ── Hydraulic conductivity ────────────────────────────────────────────
            # HC has no time dimension so we index directly into the 2-D numpy
            # array that was realised once before the main loop.
            hc_stats = {'hc_at_peak': HC_DATA[yi, xi]}

            all_results.append({**flood_stats, **mean_sm_stats, **hc_stats})

        print(f"  Took: {round(time.time() - start_time, 2)}s")

        # ── Discard SM data before loading the next year ──────────────────────────
        # Explicitly deleting both the iris cube and the numpy array ensures the
        # memory is released before the next iteration loads a new file.
        # gc.collect() forces immediate reclamation rather than waiting for the
        # next GC cycle, which matters when files are large (~hundreds of MB).
        del sm, sm_data
        gc.collect()

    results_df = pd.DataFrame(all_results)
    return results_df


## Loop through all catchments
for catchment_num in all_catchments:
    catchment_name = CATCHMENT_LOOKUP_DICT[catchment_num]
    fp = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/EventDetails/Catchment_{catchment_num}/{catchment_name}.pkl"
    flood_fp = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{catchment_num}/Ens01_{catchment_num}/10cm/flooded_area_5km_total_Ens01_{catchment_num}_10cm.nc"
        
    if os.path.isfile(fp) and os.path.isfile(flood_fp):
        if catchment_num != '23':
            print(f"{catchment_num} {catchment_name}")
            results_df = stage2_procesing(catchment_num)
            
            out_fp = f"{OUT_DIR}/Catchment_{catchment_num}.pkl"
            results_df.to_pickle(out_fp)
            del results_df
            gc.collect()
            