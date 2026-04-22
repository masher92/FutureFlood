import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt
import numpy as np
import cartopy.crs as ccrs
import gc
from shapely.ops import unary_union
from shapely.geometry import box, MultiPolygon
import iris
from shapely.geometry import box, MultiPolygon

from functions import get_rainfall_cube, mask_to_polygons

#### NEED TO CHECK IF THIS WOULD BREAK THE WAY THIS WORKS ELSEWHERE

def get_rainfall_cube_subsection(yr, ENS_NUM, RAINFALLDIR, start_idx=None, stop_idx=None):
    """
    Each monthly file contains 30*24 = 720 timesteps (30-day months, hourly data).
    If we know start_idx and stop_idx we can work out which files we actually
    need and skip loading the rest entirely.
    """
    HOURS_PER_MONTH = 30 * 24  # 720 — fixed because this is a 360-day calendar dataset

    # Build the full ordered file list with their global time index ranges
    # so we can identify which files overlap with [start_idx, stop_idx]
    all_files = []
    cumulative = 0

    # December of previous year: indices 0–719
    all_files.append({
        'path': f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr-1}1201-{yr-1}1230.nc",
        'global_start': cumulative,
        'global_end':   cumulative + HOURS_PER_MONTH
    })
    cumulative += HOURS_PER_MONTH

    # January–November of target year
    for m in range(1, 12):
        all_files.append({
            'path': f"{RAINFALLDIR}bc_pr_rcp85_land-cpm_uk_5km_{ENS_NUM}_1hr_{yr}{m:02d}01-{yr}{m:02d}30.nc",
            'global_start': cumulative,
            'global_end':   cumulative + HOURS_PER_MONTH
        })
        cumulative += HOURS_PER_MONTH

    # ── Filter to only files that overlap with [start_idx, stop_idx] ──────────
    # A file overlaps if its range intersects the requested window.
    # This typically reduces 12 file loads down to 1 or 2.
    if start_idx is not None and stop_idx is not None:
        needed = [f for f in all_files
                  if f['global_end'] > start_idx and f['global_start'] < stop_idx]
    else:
        needed = all_files

    # ── Load only the needed files ────────────────────────────────────────────
    monthly_cubes = iris.cube.CubeList()
    for f in needed:
        cube = iris.load(f['path'])[1]
        cube.attributes = {}
        monthly_cubes.append(cube)

    year_cube = monthly_cubes.concatenate_cube()

    # ── Slice to the exact requested window ───────────────────────────────────
    # Adjust indices to be relative to the first loaded file's start
    if start_idx is not None and stop_idx is not None:
        offset = needed[0]['global_start']
        return year_cube[start_idx - offset:stop_idx - offset, :, :]

    return year_cube

def get_data_at_peak_cell(cube, one_event_results, x_idx_variable, y_idx_variable):

    x_idx = getattr(one_event_results, x_idx_variable)
    y_idx = getattr(one_event_results, y_idx_variable)

    x_coord = one_event_results.x_coord
    y_coord = one_event_results.y_coord

    cube_x = cube.coord('projection_x_coordinate').points
    cube_y = cube.coord('projection_y_coordinate').points

    # Check the coordinate at the global index matches what we stored
    assert np.isclose(cube_x[x_idx], x_coord, atol=100), \
        f"x mismatch: {cube_x[x_idx]} vs {x_coord}"
    assert np.isclose(cube_y[y_idx], y_coord, atol=100), \
        f"y mismatch: {cube_y[y_idx]} vs {y_coord}"

    # Extract timeseries
    if len(cube.shape) == 2:
        cube_at_peak = cube[y_idx, x_idx]
    elif len(cube.shape) == 3:
        cube_at_peak = cube[:, y_idx, x_idx]

    return cube_at_peak

def plot_peak_check(ax, cube, catchment_poly, this_event_results, title):
    """
    Sanity check plot:
    - Full SM field at a given timestep
    - Catchment boundary
    - Peak cell marked as a scatter point
    - Neighbouring cells highlighted to confirm index alignment
    """
    
    x_coord = int(this_event_results.x_coord)
    y_coord =int(this_event_results.y_coord)
    x_idx_global = int(this_event_results.x_idx_global)
    y_idx_global = int(this_event_results.y_idx_global)
    
    sm_x = cube.coord('projection_x_coordinate').points
    sm_y = cube.coord('projection_y_coordinate').points
    dx   = sm_x[1] - sm_x[0]
    dy   = sm_y[1] - sm_y[0]

    # Full SM field
    iplt.pcolormesh(cube, axes=ax, cmap='Blues')

    # Catchment boundary
    ax.add_geometries(
        [catchment_poly], crs=ccrs.OSGB(),
        facecolor='none', edgecolor='black', linewidth=1.5)

    # Peak cell as a box (so you can see it aligns with the pcolormesh grid)
    peak_box = plt.matplotlib.patches.Rectangle(
        (x_coord - dx/2, y_coord - dy/2), dx, dy,
        linewidth=2, edgecolor='red', facecolor='red', alpha=0.1,
        transform=ccrs.OSGB())
    ax.add_patch(peak_box)

    # Centre point
    ax.scatter(x_coord, y_coord, s=20, color='red', zorder=6,
               transform=ccrs.OSGB(), label=f'Peak ({x_idx_global}, {y_idx_global})')

    # Zoom to catchment with buffer
    minx, miny, maxx, maxy = catchment_poly.bounds
    buf = 20000  # 20km buffer
    ax.set_extent([minx-buf, maxx+buf, miny-buf, maxy+buf], crs=ccrs.OSGB())

#     ax.set_title(f"Peak cell check — global idx ({x_idx_global}, {y_idx_global})\n"
#                  f"coords ({x_coord:.0f}, {y_coord:.0f})")
    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.tight_layout()


def run_spatial_diagnostics(row, event_num, sm, HC_CUBE, flood_cubes, catchment_poly,
                             ens_num, rainfall_cube_dir, depth=10):
    """
    Plots a 2x2 spatial check for a single event.
    Rainfall cube is loaded here on demand and discarded after plotting.

    Parameters
    ----------
    row               : namedtuple   — single event row from itertuples()
    event_num         : int          — event number (passed explicitly)
    sm                : iris.Cube    — soil moisture cube for this (ens, year)
    HC_CUBE           : iris.Cube    — hydraulic conductivity (static)
    flood_cubes       : dict         — pre-loaded flood cubes, keyed [ens_num][depth]
    catchment_poly    : shapely geom — catchment boundary polygon
    ens_num           : str          — ensemble member ID
    rainfall_cube_dir : str          — directory for hourly rainfall cubes
    depth             : int          — flood depth threshold to plot (10 or 30)
    """
    # Load and slice rainfall cube only when needed
    rainfall_yr_cube = get_rainfall_cube(row.year, row.ens, rainfall_cube_dir)
    rainfall_slice   = rainfall_yr_cube[row.start_idx:row.stop_idx, :, :][row.t_local, :, :]

    sm_slice    = sm[row.hydro_day_360, :, :]
    flood_slice = flood_cubes[depth]['area']
    flood_slice = flood_slice[event_num-1, :,:]
    
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 8),
                            subplot_kw={'projection': ccrs.OSGB()})
    axs = axs.flatten()

    plot_peak_check(axs[0], sm_slice,    catchment_poly, row, title='Soil moisture')
    plot_peak_check(axs[1], rainfall_slice, catchment_poly, row, title='Precipitation')
    plot_peak_check(axs[2], HC_CUBE,     catchment_poly, row, title='Hydraulic conductivity')
    plot_peak_check(axs[3], flood_slice, catchment_poly, row, title=f'Flooded area ({depth}cm)')

    fig.suptitle(
        f"Spatial diagnostics — ens={ens_num}, year={row.year}, event={event_num}\n"
        f"Peak day: {row.rainfall_peak_day.date()}",
        fontsize=12)
    plt.tight_layout()
    plt.show()

    # Discard — no reason to keep it in memory after plotting
    del rainfall_yr_cube
    gc.collect()    

def maybe_diagnose(row, event_num, condition=False, **kwargs):
    """
    Calls run_spatial_diagnostics only if condition is True.
    event_num is passed explicitly rather than read from row.
    """
    if condition:
        run_spatial_diagnostics(row, event_num=event_num, **kwargs)
        
        
def plot_cluster_check(rain, peak_slice, threshold_mask, cluster_mask,
                       row, y0, y1, x0, x1,
                       threshold_level, neighbourhood_sum, neighbourhood_size,
                       flood=None, catchment_poly=None, boundary_gdf=None,
                       buffer=2000):
    """
    Diagnostic plot to verify cluster extraction, using the same
    iris/cartopy plotting approach as plot_peak_event.
    
    Panels:
      1. Threshold mask — all cells >= threshold_value, outlined as polygons
      2. Cluster mask   — only the connected cluster containing the peak cell
      3. Flood extent   — optional, with cluster outline overlaid
    
    Parameters
    ----------
    rain             : 2D numpy array — cleaned rainfall field (NaNs for invalid)
    peak_slice       : iris Cube      — 2D spatial slice at peak timestep (for qplt)
    threshold_mask   : 2D bool array  — cells >= threshold_value
    cluster_mask     : 2D bool array  — cells in the peak-containing cluster only
    row              : namedtuple     — event row from itertuples()
    y0,y1,x0,x1     : int            — neighbourhood bounding box indices
    threshold_level  : float          — e.g. 0.8, used in title
    neighbourhood_sum: float          — precomputed neighbourhood sum, shown in title
    flood            : iris Cube      — optional 2D flood extent cube
    catchment_poly   : shapely geom   — catchment boundary for zooming
    boundary_gdf     : GeoDataFrame   — catchment boundary for plotting
    buffer           : int            — metres of padding around catchment extent
    """

    # ── Get coordinate arrays for polygon conversion ──────────────────────────
    # These are the actual projected coordinates (metres, OSGB) of each grid cell,
    # needed by mask_to_polygons to create properly georeferenced shapely polygons
    x = peak_slice.coord('projection_x_coordinate').points
    y = peak_slice.coord('projection_y_coordinate').points

    # ── Convert boolean masks to shapely polygons ─────────────────────────────
    # mask_to_polygons traces the boundary of each connected True region and
    # returns a list of shapely Polygons in OSGB coordinates — same as plot_peak_event
    threshold_polys = mask_to_polygons(threshold_mask, x, y)
    cluster_polys   = mask_to_polygons(cluster_mask,   x, y)

    # ── Also build a polygon for the neighbourhood bounding box ──────────────
    # Converts the array index bounds (y0,y1,x0,x1) into coordinate space so
    # the box is drawn correctly on the map rather than in index space
    dx          = x[1] - x[0]
    dy          = y[1] - y[0]
    box_x0      = x[x0] - dx / 2
    box_x1      = x[x1 - 1] + dx / 2
    box_y0      = y[y0] - dy / 2
    box_y1      = y[y1 - 1] + dy / 2
    neighbourhood_box = box(box_x0, box_y0, box_x1, box_y1)  # shapely.geometry.box

    # ── Layout ────────────────────────────────────────────────────────────────
    ncols = 3 if flood is not None else 2
    fig, axes = plt.subplots(
        1, ncols, figsize=(5 * ncols, 5),
        subplot_kw={'projection': ccrs.OSGB()}
    )
    axes = list(axes)

    panels = [
        ("Threshold mask\n"
         f"(>= {threshold_level*100:.0f}% of peak)",  threshold_polys, axes[0]),
        ("Peak cluster\n"
         f"neighbourhood sum = {neighbourhood_sum:.2f} mm", cluster_polys, axes[1]),
    ]

    # ── Rainfall panels ───────────────────────────────────────────────────────
    for title, polys, ax in panels:
        # Use qplt so iris handles the coordinate-aware pcolormesh
        qplt.pcolormesh(peak_slice, axes=ax, cmap='terrain')

        # Draw the mask outline polygons (threshold or cluster)
        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color='orange', linewidth=2, transform=ccrs.OSGB())

        # Mark the peak cell
        ax.scatter(row.x_coord, row.y_coord, s=50, color='red', zorder=6,
                   transform=ccrs.OSGB(),
                   label=f'Peak ({int(row.x_idx_global)},{int(row.y_idx_global)})')

        # Draw the neighbourhood bounding box
        bx, by = neighbourhood_box.exterior.xy
        ax.plot(bx, by, color='red', linewidth=1.5, linestyle='--',
                transform=ccrs.OSGB(), label=f'Neighbourhood (n={neighbourhood_size})')

        ax.set_title(title)
        ax.legend(fontsize=7, loc='lower right')

    # ── Optional flood panel ──────────────────────────────────────────────────
    if flood is not None:
        ax = axes[2]
        qplt.pcolormesh(flood, axes=ax, cmap='Blues')
        ax.set_title("Flood extent")
        ax.scatter(row.x_coord, row.y_coord, s=50, color='red',
                   zorder=6, transform=ccrs.OSGB())

        # Overlay the cluster outline on the flood panel for comparison
        for i, poly in enumerate(cluster_polys):
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color='green', linewidth=2, transform=ccrs.OSGB(),
                    label='Cluster' if i == 0 else None)
        ax.legend(fontsize=8)

    # ── Shared formatting ─────────────────────────────────────────────────────
    for ax in axes:
        if catchment_poly is not None:
            minx, miny, maxx, maxy = catchment_poly.bounds
            ax.set_extent(
                [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer],
                crs=ccrs.OSGB())
        if boundary_gdf is not None:
            boundary_gdf.boundary.plot(ax=ax, color='black')

    plt.suptitle(
        f"Cluster extraction check — ens={row.ens}, year={row.year}, event={row.event_num}\n"
        f"Peak = {row.max_precip:.2f} mm, threshold = {threshold_level*100:.0f}%"
        f" ({row.max_precip * threshold_level:.2f} mm)",
        fontsize=11
    )
    plt.tight_layout()
    plt.show()

def analyse_peak_event(row, neighbourhood_size, threshold_levels, plot=False):
    """
    Analyse rainfall clustering and flood statistics for a single peak event.
    
    Parameters
    ----------
    row               : namedtuple  — single event row from itertuples()
    neighbourhood_size: int         — radius of bounding box around peak cell (1 = 3x3)
    threshold_levels  : list[float] — list of threshold fractions to compute, e.g. [0.5, 0.8]
                                      results are stored with threshold level in the key name
    plot              : bool        — if True, produces diagnostic plots for each threshold
    
    Returns
    -------
    dict — flat results dict with keys like:
           'cluster_rain_total_t80', 'cluster_flood_10_area_t80', etc.
           plus threshold-independent keys like 'neighbourhood_rain_total'
    """

    peak_value         = row.max_precip

    # ── Load and slice rainfall cube ──────────────────────────────────────────
    event_cube = get_rainfall_cube_subsection(row.year, row.ens, rainfall_cube_dir,
                                              start_idx=row.start_idx, stop_idx=row.stop_idx)
    event_cube.data = np.where(FULL_MASK_2D, event_cube.data, np.nan)
    event_cube, x_offset, y_offset = subset_cube_to_bbox(event_cube, CATCHMENT_POLY, buffer=0)
    peak_slice = event_cube[row.t_local, :, :]

    rain = peak_slice.data.copy().astype(float)
    rain = np.where((rain == -99999) | (rain > 1e19), np.nan, rain)

    # ── Pre-extract all flood arrays ──────────────────────────────────────────
    # Extract all depth/variable combinations once, outside the threshold loop,
    # so we don't repeat the indexing for every threshold level.
    # Keyed as flood_arrays[(depth, var)], shape (y, x)
    flood_arrays = {
        (depth, var): flood_numpy[depth][var][row.event_num_this_ens - 1, :, :]
        for depth in [10, 30]
        for var in ['area', 'vol']
    }

    # ── Neighbourhood stats (threshold-independent) ───────────────────────────
    # The neighbourhood bounding box is fixed regardless of threshold level,
    # so we compute it once here and reuse across all thresholds.
    n  = neighbourhood_size
    y0 = max(0, row.y_idx - n);  y1 = min(rain.shape[0], row.y_idx + n + 1)
    x0 = max(0, row.x_idx - n);  x1 = min(rain.shape[1], row.x_idx + n + 1)

    results = {
        'neighbourhood_rain_total': float(np.nansum(rain[y0:y1, x0:x1])),
        'neighbourhood_rain_sum_excl_peak': float(np.nansum(rain[y0:y1, x0:x1]) - peak_value),
        'neighbourhood_n_cells': int(np.sum(~np.isnan(rain[y0:y1, x0:x1]))),
        **{
            f'neighbourhood_flood_{depth}_{var}': float(np.nansum(flood_arrays[(depth, var)][y0:y1, x0:x1]))
            for depth in [10, 30]
            for var in ['area', 'vol']
        }
    }

    # ── Per-threshold stats ───────────────────────────────────────────────────
    # Loop over each threshold level, compute masks, cluster, and all region stats.
    # Results are stored with a suffix like '_t80' for threshold_level=0.8,
    # '_t50' for 0.5, etc. — so all thresholds coexist in the same flat dict.
    for threshold_level in threshold_levels:

        t_suffix        = f"_t{int(threshold_level * 100)}"
        threshold_value = peak_value * threshold_level

        # Build threshold mask and connected cluster
        threshold_mask = (rain >= threshold_value) & (~np.isnan(rain))

        diagonal = False
        if not diagonal:
            labeled, num_clusters = label(threshold_mask)
        else:
            structure = generate_binary_structure(2, 2)
            labeled, num_clusters = label(threshold_mask, structure=structure)

        peak_label   = labeled[row.y_idx, row.x_idx]
        cluster_mask = (labeled == peak_label)

        # ── Rainfall totals for this threshold ────────────────────────────────
        results[f'threshold_rain_total{t_suffix}']  = float(np.nansum(rain[threshold_mask]))
        results[f'threshold_n_cells{t_suffix}']     = int(np.sum(threshold_mask))
        results[f'cluster_rain_total{t_suffix}']    = float(np.nansum(rain[cluster_mask]))
        results[f'cluster_n_cells{t_suffix}']       = int(np.sum(cluster_mask))

        # ── Flood totals for this threshold — all depth/variable combinations ─
        # Apply the same spatial masks to each flood array.
        # threshold_mask and cluster_mask are in the same (y, x) grid space as
        # the flood arrays (both subsetted to the catchment bbox), so masking
        # is valid without any re-indexing.
        for depth in [10, 30]:
            for var in ['area', 'vol']:
                fa = flood_arrays[(depth, var)]
                results[f'threshold_flood_{depth}_{var}{t_suffix}'] = float(np.nansum(fa[threshold_mask]))
                results[f'cluster_flood_{depth}_{var}{t_suffix}']   = float(np.nansum(fa[cluster_mask]))

        if plot:
            plot_cluster_check(
                rain=rain,
                peak_slice=peak_slice,
                threshold_mask=threshold_mask,
                cluster_mask=cluster_mask,
                row=row,
                y0=y0, y1=y1, x0=x0, x1=x1,
                threshold_level=threshold_level,
                neighbourhood_sum=results['neighbourhood_rain_sum_excl_peak'],
                neighbourhood_size=neighbourhood_size,
                flood=flood_cubes[10]['area'][row.event_num_this_ens - 1],
                catchment_poly=CATCHMENT_POLY,
                boundary_gdf=boundary_gdf,
            )

    return results
       