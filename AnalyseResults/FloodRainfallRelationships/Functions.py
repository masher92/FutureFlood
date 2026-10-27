import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.gridspec as gridspec
import numpy as np

def plot_depth_by_intensity_for_timeslices(df):

    binned = (df.groupby(["geo_region", "period", "intensity_bin"],observed=True)["10cm_area"].agg(["mean", "median", "count"])
              .reset_index())

    ### Set colors
    colors = plt.cm.plasma(np.linspace(0, 1, 3))
    timeperiod_colors = dict(zip(reversed(np.unique(df['period'])), colors))

    # fig, axes = plt.subplots(nrows=int(np.ceil(len(region_order) / 3)),ncols=3,figsize=(15, 4 * int(np.ceil(len(region_order) / 3))))
    fig, axes = plt.subplots(5, 3, figsize=(12, 15), sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, region in zip(axes, region_order):

        df = binned[binned["geo_region"] == region]

        for period, group in df.groupby("period",observed=False):

            ax.plot(group["intensity_bin"].astype(str),group["mean"],marker="o",label=period, color = timeperiod_colors[period])

        # ax.set_xlabel("Maximum precipitation")
        ax.set_ylabel("Mean 10cm area")
        ax.set_title(region)

        ax.legend()
        ax.tick_params(labelrotation=45)

    for j in range(len(region_order), 5 * 3):
        axes[j].axis("off")

    fig.suptitle("Changes to Mean per-event flooded area across intensity bins in the future climate");
    fig.tight_layout();
    
def add_geometries(rainfall_events_all_df, CATCHMENTS):
    # Get shape file
    regions_shp = "/scratch/hydro5/users/la17355/FUTURE-FLOOD/Results/Pluvial/v4/#ss/plots/most_exposed_events_visualisation/by_region/ukcp18-uk-land-region-hires.shp"
    regions_df = gpd.read_file(regions_shp)

    # Create columns on regions_df and catchments_df (CATCHMENTS) with each of their geometries
    regions_df["region_geometry"] = regions_df.geometry
    CATCHMENTS["catchment_geometry"] = CATCHMENTS.geometry

    # Convert events dataframe to geodataframe
    rainfall_events_all_df = gpd.GeoDataFrame(rainfall_events_all_df,
                            geometry=gpd.points_from_xy(rainfall_events_all_df["x_coord"], rainfall_events_all_df["y_coord"]), 
                                              crs=regions_df.crs)

    # Remove old sjoin columns if this dataframe has been joined before
    rainfall_events_all_df = rainfall_events_all_df.drop(
        columns=["index_right", "geo_region", "region_geometry", "HA_NAME", "catchment_geometry"], errors="ignore")

    # Join events dataframe with the regions geodata
    rainfall_events_all_df = gpd.sjoin(rainfall_events_all_df, regions_df[["geo_region", "region_geometry", "geometry"]],
        how="left", predicate="within")

    # Drop index_right before the next sjoin, or geopandas will error on the clash
    rainfall_events_all_df = rainfall_events_all_df.drop(columns=["index_right"], errors="ignore")

    # Join events dataframe with the catchments geodata
    rainfall_events_all_df = gpd.sjoin(rainfall_events_all_df, CATCHMENTS[["HA_NAME", "catchment_geometry", "geometry"]],
        how="left", predicate="within")

    rainfall_events_all_df = rainfall_events_all_df.drop(columns=["index_right"], errors="ignore")
    return rainfall_events_all_df

def national_and_regional_plot(df, region_col, region_list, value_col, group_col, group_values):
    colors = plt.cm.viridis(np.linspace(0, 1, len(group_values)))
    color_map = dict(zip(group_values, colors))

    def _plot(ax, sub, title):
        summary = sub.groupby([group_col, "intensity_bin"], observed=True)[value_col].mean().unstack()
        # summary =summary.iloc[:,:6]
        for g in group_values:
            if g in summary.index:
                ax.plot(summary.columns, summary.loc[g], marker="o", linewidth=4,
                         markersize=5, color=color_map[g], label=g)
        ax.set_title(title, fontsize=20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig = plt.figure(figsize=(14, 14))
    n_regions = len(region_list)
    nrows = int(np.ceil(n_regions / 3)) + 1
    gs = gridspec.GridSpec(nrows, 3, height_ratios=[2] + [1]*(nrows-1), hspace=0.5, wspace=0.3)

    ax_national = fig.add_subplot(gs[0, :])
    _plot(ax_national, df, "National")

    for i, region in enumerate(region_list):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row + 1, col], sharex=ax_national)
        _plot(ax, df[df[region_col] == region], region)

    handles, labels = ax_national.get_legend_handles_labels()
    fig.legend(handles, labels, title=group_col.title(), loc="upper center",
               bbox_to_anchor=(0.5, 0.95), ncol=len(group_values))
    # return fig
