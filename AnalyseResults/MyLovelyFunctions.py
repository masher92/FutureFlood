import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from matplotlib.colors import PowerNorm, Normalize
import seaborn as sns
from matplotlib.cm import ScalarMappable



def regional_period_summary(df,variables,region_col="geo_region",year_col="start_year",start_period=(1990, 2020),
                            future_period=(2050, 2080),):
    """
    Calculate regional means for two periods and the change between them.

    Parameters
    ----------
    df : pandas.DataFrame
        Event dataframe.

    variables : list of str
        Columns for which to calculate the mean.

    region_col : str
        Column containing region names.

    year_col : str
        Column containing event year.

    start_period : tuple
        (first_year, last_year) for baseline period.

    future_period : tuple
        (first_year, last_year) for future period.

    Returns
    -------
    pandas.DataFrame
        One row per region with baseline, future, absolute change
        and percentage change for each variable.
    """

    df = df.copy()

    # Keep only the two periods
    baseline_mask = df[year_col].between(*start_period)
    future_mask = df[year_col].between(*future_period)

    baseline = df[baseline_mask]
    future = df[future_mask]

    # Mean of requested variables by region
    baseline_means = baseline.groupby(region_col, observed=True)[variables].mean()

    future_means = (future.groupby(region_col, observed=True)[variables].mean())

    # Rename columns
    baseline_means = baseline_means.add_suffix("_1990_2020")
    future_means = future_means.add_suffix("_2050_2080")

    # Combine
    result = baseline_means.join(future_means,how="outer")

    # Calculate changes
    for var in variables:

        result[f"{var}_change"] = (result[f"{var}_2050_2080"] - result[f"{var}_1990_2020"])
        result[f"{var}_pct_change"] = (result[f"{var}_change"]/ result[f"{var}_2050_2080"]* 100)

    # Number of events
    baseline_counts = (baseline.groupby(region_col, observed=True).size().rename("n_events_1990_2020"))
    future_counts = (future.groupby(region_col, observed=True).size().rename("n_events_2050_2080"))

    result = result.join(baseline_counts, how="outer")
    result = result.join(future_counts, how="outer")

    result["n_events_change"] = (result["n_events_2050_2080"]- result["n_events_1990_2020"])
    result["n_events_pct_change"] = (result["n_events_change"]/ result["n_events_1990_2020"]* 100)
    
    # Mean number of events per year
    baseline_counts_mean = (baseline.groupby(region_col, observed=True).size() / (start_period[1] - start_period[0] + 1)).rename("mean_events_per_year_1990_2020")
    future_counts_mean = (future.groupby(region_col, observed=True).size() / (future_period[1] - future_period[0] + 1)).rename("mean_events_per_year_2050_2080")

    result = result.join(baseline_counts_mean, how="outer")
    result = result.join(future_counts_mean, how="outer")

    result["mean_events_per_year_change"] = (result["mean_events_per_year_2050_2080"] - result["mean_events_per_year_1990_2020"])
    result["mean_events_per_year_pct_change"] = (result["mean_events_per_year_change"] / result["mean_events_per_year_1990_2020"] * 100)
    
    return result.reset_index()

def regional_trends(df, region_col, value_col, agg_func, label):
    """Compute annual regional series, then fit trend + % change per region."""
    if agg_func == "count":
        annual = df.groupby([region_col, "start_year"], observed=True).size()
    else:
        annual = df.groupby([region_col, "start_year"], observed=True)[value_col].mean()
    
    annual = annual.rename(label).reset_index()
    
    results = (annual.groupby(region_col, observed=True)
        .apply(lambda g: fit_trend(g["start_year"].values, g[label].values), include_groups=False)
        .reset_index())
    results["variable"] = label
    return results    

    
def fit_trend(years, values):
    """Return slope, intercept, baseline (first 10yr mean), and % change over the period."""
    mask = ~np.isnan(values)
    years, values = years[mask], values[mask]
    if len(years) < 5:
        return pd.Series({"slope": np.nan, "pct_change": np.nan, "baseline": np.nan, "p_value": np.nan})
    
    slope, intercept, r_value, p_value, std_err = linregress(years, values)
    
    # baseline = mean of first 10 years of data (more robust than a single fitted point)
    baseline_years = years <= years.min() + 20
    baseline = values[baseline_years].mean()
    
    total_change = slope * (years.max() - years.min())
    pct_change = (total_change / baseline) * 100 if baseline != 0 else np.nan
    
    return pd.Series({
        "slope": slope,
        "baseline": baseline,
        "pct_change": pct_change,
        "p_value": p_value})


def make_subplot(axs, variable, y_label, title, trend_rounding, trend_unit ):


    slope, intercept, r_value, p_value, std_err = linregress(variable.index,variable.values)
    trend = intercept + slope * variable.index
    if p_value < 0.05:
        p_value = '<0.05'
        
    rolling_variable  =variable.rolling(window=10,center=True, min_periods=5).mean()        

    axs.plot(variable.index,variable.values,marker="o",label="Annual mean", linewidth=3, markersize=3, 
                color ='royalblue')
    axs.plot(rolling_variable.index,rolling_variable.values,label="10 year rolling mean", linewidth=3,
                                        color ='darkorange')
    axs.plot(variable.index, trend, linestyle="--", label=f"Trend ({round(slope, trend_rounding)} {trend_unit}, \n p={p_value})", color="green")
    axs.set_ylabel(y_label, fontsize=20)
    axs.set_title(title, fontsize=20)
    axs.legend(fontsize=14);
    
    axs.tick_params(axis='both', which='major', labelsize=12)

def make_spatial_change_plot(summary_gdf, variable, title, unit):
    fig, axes = plt.subplots(1, 2,figsize=(10, 6))
    axes=axes.flatten()
    # Number of events
    summary_gdf.plot(column=f"{variable}_1990_2020",ax=axes[0],cmap="YlOrRd",edgecolor="black",
                           linewidth=0.5,legend=True, legend_kwds={"label": "n"})
    #                  norm=PowerNorm(gamma=0.3)
    axes[0].set_title(f"{title} \n1990–2020", fontsize=18)
    axes[0].set_axis_off()

    cbar = axes[0].get_figure().axes[-1]
    cbar.tick_params(labelsize=15)
    cbar.set_ylabel(unit, fontsize=14)

    # Max precipitation
    summary_gdf.plot(column=f"{variable}_2050_2080",ax=axes[1],cmap="YlOrRd",edgecolor="black",
                           linewidth=0.5,legend=True, legend_kwds={"label": "n"})

    cbar = axes[1].get_figure().axes[-1]
    cbar.tick_params(labelsize=15)
    cbar.set_ylabel(unit, fontsize=14)

    axes[1].set_title(f"{title} \n2050–2080", fontsize=18)
    axes[1].set_axis_off()

    fig.tight_layout()
    
def make_HC_change_plot(summary_HC, variable, title):

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # HC layout
    hc_layout = {"A": (0, 0),"B": (0, 1),"C": (1, 0),"D": (1, 1)}

    # Columns for each period
    columns = [(f"{variable}_1990_2020", "1990–2020"),(f"{variable}_2050_2080", "2050–2080")]
    cmap = plt.cm.YlOrRd

    for ax, (column, period) in zip(axes, columns):

        # Own colour scale for this period
        vmin = summary_HC[column].min()
        vmax = summary_HC[column].max()

        norm = Normalize(vmin=vmin, vmax=vmax)

        for _, row in summary_HC.iterrows():

            hc = row["HC_status"]

            if hc not in hc_layout:
                continue

            r, c = hc_layout[hc]
            value = row[column]

            x = c
            y = 1 - r

            rect = plt.Rectangle((x, y),1,1,facecolor=cmap(norm(value)),edgecolor="black",linewidth=1.5)

            ax.add_patch(rect)

            ax.text(x + 0.5,y + 0.5,hc,ha="center",va="center",fontsize=22,fontweight="bold")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\n{period}", fontsize=18)

        # Colourbar specific to this panel
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm,ax=ax,fraction=0.046,pad=0.04)

        cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()

    return fig, axes
    
    
def make_plots(variable, df, axs, row_num, region):
    
    if variable == 'count':
        annual = (df.groupby("start_year", observed=True).size())
    else:
        annual = (df.groupby("start_year", observed=True)[variable].mean())
    
    rolling= annual.rolling(window=5,center=True, min_periods=5).mean()
    
    # --------------------------------------------------
    # Column 1: Event frequency
    # --------------------------------------------------

    slope, intercept, r_value, p_value, std_err = linregress(annual.index,annual.values)
    trend = intercept + slope * annual.index
    axs[row_num].plot(rolling.index,rolling,linewidth=4, color='darkorange')
    axs[row_num].plot(annual.index,annual.values , linewidth=1, markersize=2, color ='royalblue')
    axs[row_num].plot(annual.index,trend,linestyle="--", linewidth=3, color='green',
                         label=f"Trend: {slope:+.2f} events/yr, \n p={p_value:.3f}")
    axs[row_num].set_ylabel(region, fontsize=15)
    axs[row_num].grid(alpha=0.2)
    axs[row_num].legend()    
    
    
def plot_all_with_regions(rainfall_events_all_df, variable, title, regions):
    

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(1 + int(np.ceil(13 / 4)), 4)

    ax_all = fig.add_subplot(gs[0, :])
    make_plots(variable, rainfall_events_all_df, [ax_all], 0, "All")

    for row_num, region in enumerate(regions):
        row = row_num // 4 + 1
        col = row_num % 4
        ax = fig.add_subplot(gs[row, col])

        one_region = rainfall_events_all_df[rainfall_events_all_df["geo_region"] == region]
        make_plots(variable,one_region, [ax], 0, region)

    fig.suptitle(title)  
    fig.tight_layout()   
    
    
# def plot_all_with_regions(variable, title ):
    

#     fig = plt.figure(figsize=(13, 12))
#     gs = fig.add_gridspec(1 + int(np.ceil(13 / 4)), 4)

#     ax_all = fig.add_subplot(gs[0, :])
#     make_plots(variable, rainfall_events_all_df, [ax_all], 0, "All")

#     for row_num, region in enumerate(regions):
#         row = row_num // 4 + 1
#         col = row_num % 4
#         ax = fig.add_subplot(gs[row, col])

#         one_region = rainfall_events_all_df[rainfall_events_all_df["geo_region"] == region]
#         make_plots(variable,one_region, [ax], 0, region)

#     fig.suptitle(title)  
#     fig.tight_layout()        