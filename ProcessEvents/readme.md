# Rainfall Event Detail Extraction

Processes UKCP18 bias-corrected hourly rainfall data across catchments and ensemble members to extract spatial and temporal characteristics of rainfall events that exceed a fixed intensity threshold.

---

## Overview

For each catchment and ensemble member combination, the script:

1. Loads pre-identified rainfall events from CSV files
2. Loads the corresponding hourly rainfall cube and masks it to the catchment boundary
3. Locates the peak precipitation cell within each event
4. Extracts the temporal profile of rainfall at that peak cell
5. Saves results as a pickle file per catchment

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` / `pandas` | Array operations and data handling |
| `geopandas` / `shapely` | Catchment boundary loading and spatial masking |
| `cftime` | Handling 360-day calendar dates from UKCP18 data |
| `functions_clean` | Project-specific helper functions (see below) |

Install dependencies via conda or pip as appropriate for your HPC environment.

---

## Directory Structure

The script expects the following directory layout:

```
FutureFlood/
├── Data/
│   ├── CatchmentShapefiles/
│   │   └── hyd_areas_GB_with_subcatchments.shp   # Catchment boundary polygons
│   └── EventDetails/                              # Output directory (created automatically)
│       └── Catchment_{num}/
│           └── {catchment_name}.pkl

/scratch/hydro4/users/la17355/FUTURE-FLOOD/
├── UKCP_rainfall_events/
│   └── fixed_threshold_30mm_with_volume/
│       └── {catchment_name}_{ens_num}_full_events_with_event_nums.csv
└── Data/CEH_catchments/.../
    └── hyd_areas_GB_with_subcatchments_no_spaces.csv  # Catchment number/name lookup

/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/
└── {ens_num}/                                         # Per-ensemble rainfall cubes
```

---

## Configuration

Edit the constants at the top of the script before running:

| Variable | Description |
|---|---|
| `MOLLY_DIR_FF` | Root directory for FutureFlood project files |
| `RAINFALL_CSV_DIR` | Directory containing per-catchment event CSVs |
| `OUT_DIR` | Output directory for results |
| `ENSEMBLE_MEMBERS` | List of UKCP18 ensemble member IDs to process |

---

## Usage

### Running the script

```bash
python extract_event_details.py
```

The script automatically detects already-completed catchments by checking for existing output directories under `OUT_DIR`, so it is safe to re-run after an interruption.

Catchment 105 is explicitly excluded (see `catchments_to_run`) — update this as needed.

### Running with parallelisation

To process multiple catchments simultaneously, wrap the outer loop using `ProcessPoolExecutor` (see the parallel version of this script). Set `N_WORKERS` conservatively — each worker loads full rainfall cubes into memory, so memory is typically the binding constraint rather than CPU count. Start with 2–4 workers and monitor with `htop`.

```bash
python extract_event_details_parallel.py
```

---

## Output

Results are saved as pandas DataFrames serialised to pickle format:

```
OUT_DIR/
└── Catchment_{num}/
    └── {catchment_name}.pkl
```

Each row in the DataFrame corresponds to one rainfall event and contains:

| Field | Description |
|---|---|
| `ens` | Ensemble member ID |
| `year` | Year of the event |
| `start_idx` / `stop_idx` | Time index bounds of the event within the annual cube |
| `rainfall_peak_day` | Date of peak rainfall (360-day calendar) |
| `t_global` | Global time index of the peak timestep |
| *(spatial fields)* | Location of peak precipitation cell (from `find_max_precip_location`) |
| *(temporal fields)* | Temporal profile metrics (from `find_temporal_profile`) |

Intermediate per-ensemble `.pkl` files are written during processing as a checkpoint. These are deleted automatically once the combined catchment file is successfully produced.

---

## Key Functions (from `functions_clean`)

| Function | Description |
|---|---|
| `get_rainfall_event_details(events_df, event_num)` | Returns metadata for a single event: year, start/stop indices, peak precipitation |
| `get_rainfall_cube(year, ens_num, cube_dir)` | Loads the hourly rainfall iris cube for a given year and ensemble member |
| `mask_cube_with_catchment_full_grid(cube_slice, polygon, method)` | Creates a 2D boolean mask over the full grid for a catchment polygon |
| `subset_cube_to_bbox(cube, polygon, buffer)` | Trims the cube spatially to the catchment bounding box, returning offsets |
| `find_max_precip_location(cube, start, stop, ...)` | Finds the grid cell with peak accumulated precipitation over the event window |
| `get_data_at_peak_cell(cube, event_results)` | Extracts the full time series at the peak cell |
| `find_temporal_profile(rainfall_series, event_results, plot)` | Characterises the temporal shape of rainfall at the peak cell |

---

## Performance Notes

- A single 2D catchment mask (`FULL_MASK_2D`) is computed once per ensemble member using an arbitrary year's cube (the grid is time-invariant) and reused across all years.
- Event metadata (`get_rainfall_event_details`) is pre-computed into a dictionary cache (`event_details_cache`) before the year/event loops to avoid redundant computation.
- Annual cubes are spatially subsetted to the catchment bounding box before event processing to reduce memory use and indexing overhead.
- Cubes are explicitly deleted and garbage collected after each year to manage memory on shared HPC nodes.

---

## Resumability

The script checks `OUT_DIR` for existing `Catchment_{num}` directories at startup and skips any catchment that already has output. If a run is interrupted mid-catchment, delete that catchment's directory before re-running to ensure clean output.
