def extract_event_number(filename):
    """
    Extract event number from filenames like:
    res_23_1993_1_Ens07_binary_10cm.tif -> 1
    res_23_1993_123_Ens07_binary_10cm.tif -> 123
    """
    match = re.search(r'_(\d+)_Ens\d+_(?:binary|filtered)_', filename)
    if match:
        return int(match.group(1))
    return None


def iter_event_tifs(ha_num, data_kind, ensembles=None, thresholds=None):
    """
    Yield metadata for every event tif in:
    tifs/EnsXX_<HA_NUM>/<data_kind>/10cm/*.tif
    tifs/EnsXX_<HA_NUM>/<data_kind>/30cm/*.tif
    """
    if data_kind not in {"binary", "filtered"}:
        raise ValueError(f"Unsupported data_kind={data_kind}")

    ensembles = ensembles or ENSEMBLE_MEMBERS
    thresholds = thresholds or THRESHOLDS
    ha_num = str(ha_num)

    for ens in ensembles:
        ens_name = f"Ens{ens}_{ha_num}"

        for thr in thresholds:
            tif_glob = os.path.join(TIFS_DIR, ens_name, data_kind, thr, "*.tif")
            tif_paths = sorted(glob.glob(tif_glob))

            if not tif_paths:
                print(f"[WARN] No files found: {os.path.dirname(tif_glob)}")
                continue

            for tif_path in tif_paths:
                event_num = extract_event_number(os.path.basename(tif_path))
                if event_num is None:
                    print(f"[WARN] Could not extract event number from: {os.path.basename(tif_path)}")
                    continue
                
                yield {
                    "ha_num": ha_num,
                    "ensemble": ens_name,
                    "threshold": thr,
                    "data_kind": data_kind,
                    "path": tif_path,
                    "event_num": event_num,
                }


def process_single_event_area(binary_file, x5, y5, nx, ny, qa_out_tif=None):
    """
    Process a single binary flood tif and return flooded area per 5km grid cell.
    Maps 30m pixels directly to the original 5km grid using coordinates.
    """
    # Open flood raster
    flood = rxr.open_rasterio(
        binary_file,
        chunks={"x": 2000, "y": 2000}
    ).squeeze()

    if flood.rio.crs is None or flood.rio.crs.to_string() != "EPSG:27700":
        flood = flood.rio.reproject("EPSG:27700")

    # Identify valid pixels
    valid = flood.notnull()

    # Get flood pixel coordinates and values
    flood_x = flood.x.values
    flood_y = flood.y.values
    
    # Create meshgrid of flood coordinates
    xx, yy = np.meshgrid(flood_x, flood_y)
    
    # Flatten coordinates and values
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    flood_flat = flood.values.ravel()
    valid_flat = valid.values.ravel()
    
    # Only keep valid, flooded pixels
    mask = valid_flat & (flood_flat > 0)
    x_flood = x_flat[mask]
    y_flood = y_flat[mask]
    
    if x_flood.size == 0:
        return np.zeros((ny, nx), dtype=np.float32)

    # Map flood-pixel centers to 5km cell indices using cell edges.
    x5_arr = np.asarray(x5.values)
    y5_arr = np.asarray(y5.values)

    dx5 = float(abs(x5_arr[1] - x5_arr[0]))
    dy5 = float(abs(y5_arr[1] - y5_arr[0]))

    if x5_arr[1] > x5_arr[0]:
        xmin_edge = float(np.min(x5_arr) - dx5 / 2.0)
        ix = np.floor((x_flood - xmin_edge) / dx5).astype(np.int64)
    else:
        xmax_edge = float(np.max(x5_arr) + dx5 / 2.0)
        ix = np.floor((xmax_edge - x_flood) / dx5).astype(np.int64)

    if y5_arr[1] > y5_arr[0]:
        ymin_edge = float(np.min(y5_arr) - dy5 / 2.0)
        iy = np.floor((y_flood - ymin_edge) / dy5).astype(np.int64)
    else:
        ymax_edge = float(np.max(y5_arr) + dy5 / 2.0)
        iy = np.floor((ymax_edge - y_flood) / dy5).astype(np.int64)

    in_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[in_grid]
    iy = iy[in_grid]

    if ix.size == 0:
        return np.zeros((ny, nx), dtype=np.float32)
    
    # Convert 2D indices to 1D linear indices
    linear_idx = iy * nx + ix

    if qa_out_tif is not None:
        # Build QA raster in original 30m grid:
        # value = 5km linear cell index for flooded pixels, -1 elsewhere.
        flooded_flat_idx = np.where(mask)[0]
        flooded_flat_idx_in_grid = flooded_flat_idx[in_grid]
        qa_flat = np.full(flood_flat.shape, -1, dtype=np.int32)
        qa_flat[flooded_flat_idx_in_grid] = linear_idx.astype(np.int32)
        qa_arr = qa_flat.reshape(flood.shape)

        qa_da = xr.DataArray(qa_arr, coords=flood.coords, dims=flood.dims)
        qa_da = qa_da.rio.write_crs(flood.rio.crs)
        qa_da.rio.write_transform(flood.rio.transform(), inplace=True)

        os.makedirs(os.path.dirname(qa_out_tif), exist_ok=True)
        qa_da.rio.to_raster(qa_out_tif)
    
    # Count flooded pixels per grid cell
    counts_flat = np.bincount(linear_idx, minlength=nx * ny)
    counts = counts_flat.reshape(ny, nx)
    
    # Convert flooded-pixel counts to total flooded area using actual raster resolution.
    xres, yres = flood.rio.resolution()
    pixel_area_km2 = (abs(xres) * abs(yres)) / 1e6
    flood_area = counts.astype(np.float32) * pixel_area_km2
    
    return flood_area


def process_single_event_volume(depth_file, x5, y5, nx, ny):
    """
    Process a single depth flood tif and return flooded volume per 5km grid cell.
    Volume is computed as sum(depth_m * pixel_area_m2) over pixels within each 5km cell.
    """
    depth = rxr.open_rasterio(
        depth_file,
        chunks={"x": 2000, "y": 2000}
    ).squeeze()

    if depth.rio.crs is None or depth.rio.crs.to_string() != "EPSG:27700":
        depth = depth.rio.reproject("EPSG:27700")

    valid = depth.notnull()

    depth_x = depth.x.values
    depth_y = depth.y.values
    xx, yy = np.meshgrid(depth_x, depth_y)

    x_flat = xx.ravel()
    y_flat = yy.ravel()
    depth_flat = depth.values.ravel()
    valid_flat = valid.values.ravel()

    # Only include valid, positive depths in the volume sum.
    mask = valid_flat & (depth_flat > 0)
    x_depth = x_flat[mask]
    y_depth = y_flat[mask]
    depth_vals = depth_flat[mask].astype(np.float64)

    if x_depth.size == 0:
        return np.zeros((ny, nx), dtype=np.float32)

    x5_arr = np.asarray(x5.values)
    y5_arr = np.asarray(y5.values)

    dx5 = float(abs(x5_arr[1] - x5_arr[0]))
    dy5 = float(abs(y5_arr[1] - y5_arr[0]))

    if x5_arr[1] > x5_arr[0]:
        xmin_edge = float(np.min(x5_arr) - dx5 / 2.0)
        ix = np.floor((x_depth - xmin_edge) / dx5).astype(np.int64)
    else:
        xmax_edge = float(np.max(x5_arr) + dx5 / 2.0)
        ix = np.floor((xmax_edge - x_depth) / dx5).astype(np.int64)

    if y5_arr[1] > y5_arr[0]:
        ymin_edge = float(np.min(y5_arr) - dy5 / 2.0)
        iy = np.floor((y_depth - ymin_edge) / dy5).astype(np.int64)
    else:
        ymax_edge = float(np.max(y5_arr) + dy5 / 2.0)
        iy = np.floor((ymax_edge - y_depth) / dy5).astype(np.int64)

    in_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[in_grid]
    iy = iy[in_grid]
    depth_vals = depth_vals[in_grid]

    if ix.size == 0:
        return np.zeros((ny, nx), dtype=np.float32)

    linear_idx = iy * nx + ix

    xres, yres = depth.rio.resolution()
    pixel_area_m2 = abs(xres) * abs(yres)

    # Depth rasters are in meters, so depth[m] * area[m2] -> volume[m3].
    contrib_m3 = depth_vals * pixel_area_m2
    volume_flat = np.bincount(linear_idx, weights=contrib_m3, minlength=nx * ny)
    flood_volume = volume_flat.reshape(ny, nx).astype(np.float32)

    return flood_volume
    
import os
import glob
import re
import argparse
import xarray as xr
import rioxarray as rxr
import numpy as np
from rasterio.transform import from_bounds

# -----------------------------
# CONFIG
# -----------------------------
ROOT_DIR = "/scratch/hydro5/users/la17355/FUTURE-FLOOD/Results/Pluvial/v4"
# OUTPUT_DIR = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{HA_NUM}"

TIFS_DIR = os.path.join(ROOT_DIR, "tifs")

# Your 12 ensembles
ENSEMBLE_MEMBERS = ["01", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "15"]

# Only the two thresholds you care about
THRESHOLDS = ["10cm", "30cm"]

# QA output: write 30m geotiffs showing assigned 5km cell id per flooded pixel.
WRITE_QA_GRID_INDEX_TIF = False
QA_MAX_EVENTS_PER_THRESHOLD = 1

# Input grid
GRID_5KM_FILE = "/scratch/hydro4/users/la17355/FUTURE-FLOOD/UKCP_rainfall/5km/Ens_01/bc_pr_rcp85_land-cpm_uk_5km_01_1hr_19901201-19911130.nc"
# GRID_5KM_FILE = "/scratch/hydro5/users/ld14116/SDM_bias_correction/Hourly/01/bc_pr_rcp85_land-cpm_uk_5km_01_1hr_20801101-20801130.nc"

files = os.listdir("/scratch/hydro5/users/la17355/FUTURE-FLOOD/Results/Pluvial/v4/tifs/")

# Extract everything after 'Ens15_'
catchment_numbers = set()
for f in files:
    match = re.search(r'Ens15_(.+)', f)
    if match:
        catchment_numbers.add(match.group(1))

# -----------------------------
# FILTER TO ONLY INCOMPLETE CATCHMENTS
# -----------------------------
def all_outputs_exist(ha_num):
    out_dir_base = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{ha_num}"
    for ens in ENSEMBLE_MEMBERS:
        for thr in THRESHOLDS:
            out_dir = os.path.join(out_dir_base, f"Ens{ens}_{ha_num}", thr)
            for kind in ("area", "volume"):
                fname = f"flooded_{kind}_5km_total_Ens{ens}_{ha_num}_{thr}.nc"
                if not os.path.exists(os.path.join(out_dir, fname)):
                    return False
    return True

catchments_to_skip = {c for c in catchment_numbers if all_outputs_exist(c)}
catchments_to_run = catchment_numbers - catchments_to_skip

print(f"{len(catchments_to_skip)} catchments already complete, skipping.")
print(f"{len(catchments_to_run)} catchments to process: {sorted(catchments_to_run)}")


for HA_NUM in catchments_to_run:
    print(f"Running for {HA_NUM}")
    OUT_DIR = f"/scratch/hydro4/users/kv25483/FutureFlood/Data/PluvialResults/5km_total/Catchment_{HA_NUM}"
    print(f"Outputs to be stored in {OUT_DIR}")
    
    # -----------------------------
    # OPEN 5km GRID
    # -----------------------------
    print(f"Loading 5km grid from {GRID_5KM_FILE}...")
    ds = xr.open_dataset(GRID_5KM_FILE)

    x5 = ds["projection_x_coordinate"]
    y5 = ds["projection_y_coordinate"]

    nx = x5.size
    ny = y5.size

    dx = float(abs(x5[1] - x5[0]))
    dy = float(abs(y5[1] - y5[0]))

    xmin = float(x5.min() - dx/2)
    xmax = float(x5.max() + dx/2)
    ymin = float(y5.min() - dy/2)
    ymax = float(y5.max() + dy/2)

    # -----------------------------
    # CREATE GRID-ID RASTER
    # -----------------------------
    grid_ids = np.arange(nx * ny).reshape(ny, nx)

    # Use standard y, x dimension names for rioxarray compatibility
    grid_da = xr.DataArray(
        grid_ids,
        coords={"y": y5.values, "x": x5.values},
        dims=("y", "x")
    )

    grid_da = grid_da.rio.write_crs("EPSG:27700")

    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    grid_da.rio.write_transform(transform, inplace=True)

    # Export a GeoTIFF copy of the 5km grid IDs for visual QA.
    # grid_out_dir = os.path.join(OUT_DIR, "grid")
    # os.makedirs(grid_out_dir, exist_ok=True)
    # grid_tif_path = os.path.join(grid_out_dir, f"grid_id_5km_from_input_{HA_NUM}.tif")
    # print(f"Saving 5km grid GeoTIFF to {grid_tif_path}...")
    # grid_da.astype("int32").rio.to_raster(grid_tif_path)

    # -----------------------------
    # COLLECT ALL EVENT FILES
    # -----------------------------
    print(f"Scanning for binary tif files for HA_NUM={HA_NUM}...")
    binary_event_files = list(iter_event_tifs(HA_NUM, data_kind="binary"))

    print(f"Scanning for filtered depth tif files for HA_NUM={HA_NUM}...")
    filtered_event_files = list(iter_event_tifs(HA_NUM, data_kind="filtered"))

    if not binary_event_files:
        raise ValueError(f"No binary tif files found for HA_NUM={HA_NUM}")

    if not filtered_event_files:
        raise ValueError(f"No filtered depth tif files found for HA_NUM={HA_NUM}")

    print(f"Found {len(binary_event_files)} binary event files")
    print(f"Found {len(filtered_event_files)} filtered depth event files")

    # -----------------------------
    # GROUP BY ENSEMBLE
    # -----------------------------
    from collections import defaultdict
    binary_by_ensemble = defaultdict(list)
    filtered_lookup = {}

    for info in binary_event_files:
        binary_by_ensemble[info['ensemble']].append(info)

    for info in filtered_event_files:
        key = (info["ensemble"], info["threshold"], info["event_num"])
        filtered_lookup[key] = info

    print(f"Found {len(binary_by_ensemble)} ensemble members")

    # -----------------------------
    # PROCESS EACH ENSEMBLE + THRESHOLD SEPARATELY
    # -----------------------------
    for ens_name in sorted(binary_by_ensemble.keys()):
#     for ens_name in ['Ens13_106']:        
        ens_events = binary_by_ensemble[ens_name]
        ens_events.sort(key=lambda x: (x['threshold'], x['event_num']))

        print(f"\n{'='*60}")
        print(f"Processing {ens_name}: {len(ens_events)} total events")
        print(f"{'='*60}")
        
        for thr in THRESHOLDS:
            thr_events = [e for e in ens_events if e["threshold"] == thr]
            if not thr_events:
                print(f"[WARN] No events for {ens_name} threshold {thr}")
                continue

            out_dir = os.path.join(OUT_DIR, ens_name, thr)
            os.makedirs(out_dir, exist_ok=True)

            output_area_nc = os.path.join(out_dir, f"flooded_area_5km_total_{ens_name}_{thr}.nc")
            output_volume_nc = os.path.join(out_dir, f"flooded_volume_5km_total_{ens_name}_{thr}.nc")

            if os.path.exists(output_area_nc) and os.path.exists(output_volume_nc):
                print(f"[SKIP] Outputs already exist for {ens_name} | {thr}")
                continue

            print(f"\n[{ens_name} | {thr}] Processing {len(thr_events)} events")

            flood_areas = []
            flood_volumes = []
            for i, info in enumerate(thr_events):
                print(f"[{i+1}/{len(thr_events)}] Processing {os.path.basename(info['path'])} (event={info['event_num']})")

                key = (ens_name, thr, info["event_num"])
                if key not in filtered_lookup:
                    raise ValueError(
                        f"Missing filtered depth tif for {ens_name}, {thr}, event={info['event_num']}"
                    )

                filtered_info = filtered_lookup[key]

                qa_out_tif = None
                if WRITE_QA_GRID_INDEX_TIF and i < QA_MAX_EVENTS_PER_THRESHOLD:
                    print("Performing QA")
                    qa_dir = os.path.join(OUT_DIR, "qa", ens_name, thr)
                    qa_out_tif = os.path.join(
                        qa_dir,
                        f"qa_5km_cell_index_{ens_name}_{thr}_event_{info['event_num']:03d}.tif"
                    )
                    print(f"    Writing QA 30m->5km index raster: {qa_out_tif}")
                else:
                    print("Skippping QA")

                flood_area = process_single_event_area(
                    info['path'],
                    x5, y5, nx, ny,
                    qa_out_tif=qa_out_tif)

                flood_volume = process_single_event_volume(
                    filtered_info["path"],
                    x5, y5, nx, ny)

                total_km2 = float(np.sum(flood_area))
                total_m3 = float(np.sum(flood_volume))
                print(f"    Total flooded area (sum of 5km cells): {total_km2:.4f} km2")
                print(f"    Total flooded volume (sum of 5km cells): {total_m3:.2f} m3")
                flood_areas.append(flood_area)
                flood_volumes.append(flood_volume)

            flood_areas_stack = np.stack(flood_areas, axis=0)
            flood_volumes_stack = np.stack(flood_volumes, axis=0)
            event_nums = np.array([info['event_num'] for info in thr_events], dtype=np.int32)

            out_area = xr.Dataset(
                {
                    "flooded_area_5km_km2": (
                        ("event", "projection_y_coordinate", "projection_x_coordinate"),
                        flood_areas_stack
                    ),
                    "event_num": ("event", event_nums),
                },
                coords={
                    "event": event_nums,
                    "projection_x_coordinate": x5,
                    "projection_y_coordinate": y5
                }
            )

            out_volume = xr.Dataset(
                {
                    "flooded_volume_5km_m3": (
                        ("event", "projection_y_coordinate", "projection_x_coordinate"),
                        flood_volumes_stack
                    ),
                    "event_num": ("event", event_nums),
                },
                coords={
                    "event": event_nums,
                    "projection_x_coordinate": x5,
                    "projection_y_coordinate": y5
                }
            )

            for out_ds in (out_area, out_volume):
                out_ds["projection_x_coordinate"].attrs.update({
                    "standard_name": "projection_x_coordinate",
                    "long_name": "x coordinate of British National Grid projection",
                    "units": "m"
                })
                out_ds["projection_y_coordinate"].attrs.update({
                    "standard_name": "projection_y_coordinate",
                    "long_name": "y coordinate of British National Grid projection",
                    "units": "m"
                })
                out_ds["event_num"].attrs["long_name"] = "event number"
                out_ds["event"].attrs["long_name"] = "event number"

            out_area = out_area.rio.set_spatial_dims(
                x_dim="projection_x_coordinate",
                y_dim="projection_y_coordinate"
            )
            out_area.rio.write_transform(transform, inplace=True)
            out_area.rio.write_crs("EPSG:27700", inplace=True)
            out_area.rio.write_coordinate_system(inplace=True)

            out_volume = out_volume.rio.set_spatial_dims(
                x_dim="projection_x_coordinate",
                y_dim="projection_y_coordinate"
            )
            out_volume.rio.write_transform(transform, inplace=True)
            out_volume.rio.write_crs("EPSG:27700", inplace=True)
            out_volume.rio.write_coordinate_system(inplace=True)

            out_area["flooded_area_5km_km2"].attrs["long_name"] = "Total flooded area per 5km grid cell"
            out_area["flooded_area_5km_km2"].attrs["units"] = "km2"
            out_volume["flooded_volume_5km_m3"].attrs["long_name"] = "Total flooded volume per 5km grid cell"
            out_volume["flooded_volume_5km_m3"].attrs["units"] = "m3"

            out_dir = os.path.join(OUT_DIR, ens_name, thr)
            os.makedirs(out_dir, exist_ok=True)

            output_area_nc = os.path.join(out_dir, f"flooded_area_5km_total_{ens_name}_{thr}.nc")
            output_volume_nc = os.path.join(out_dir, f"flooded_volume_5km_total_{ens_name}_{thr}.nc")

            print(f"Saving area to {output_area_nc}...")
            out_area.to_netcdf(output_area_nc)
            print(f"Done! Saved {len(thr_events)} events to {output_area_nc}")

            print(f"Saving volume to {output_volume_nc}...")
            out_volume.to_netcdf(output_volume_nc)
            print(f"Done! Saved {len(thr_events)} events to {output_volume_nc}")

    print(f"\n{'='*60}")
    print(f"All {len(binary_by_ensemble)} ensemble members processed!")