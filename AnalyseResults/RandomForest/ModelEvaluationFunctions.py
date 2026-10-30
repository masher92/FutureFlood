import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, cross_validate
from collections import namedtuple
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import shap
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from matplotlib.patches import Patch

def loco_test(X_full, y, drop_var, cv):
    X_reduced = X_full.drop(columns=[drop_var])
    rf_test = RandomForestRegressor(n_estimators=100, random_state=42)
    r2 = cross_val_score(rf_test, X_reduced, y, cv=cv, scoring='r2').mean()
    return r2

def loco_test_vars(X_full, y, drop_vars, cv):
    X_reduced = X_full.drop(columns=drop_vars)
    rf_test = RandomForestRegressor(n_estimators=100, random_state=42)
    r2 = cross_val_score(rf_test, X_reduced, y, cv=cv, scoring='r2').mean()
    return r2

ShapGroupResult = namedtuple("ShapGroupResult", ["summary", "shap_df", "group_shap_df", "X_shap"])
def SHAP_values_with_groups(rf, X_test, individual_variables, variable_groups, SEED, subset_size=None):
    if subset_size is not None:
        X_test = X_test.sample(min(subset_size, len(X_test)), random_state=SEED)

    shap_values = shap.TreeExplainer(rf).shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)

    # Per-sample SHAP for each group: sum the members' signed values
    group_shap = {}
    for group_name, variables in variable_groups.items():
        cols = [c for c in variables if c in shap_df.columns]
        if cols:
            group_shap[group_name] = shap_df[cols].sum(axis=1)
    group_shap_df = pd.DataFrame(group_shap)

    # Combined per-sample table: individuals + groups
    combined = pd.concat([shap_df[individual_variables], group_shap_df], axis=1)

    summary = pd.DataFrame({
        "type": ["individual"] * len(individual_variables) + ["group"] * group_shap_df.shape[1],
        "predictor": combined.columns,
        "SHAP_mean_abs": combined.abs().mean().values,
        "SHAP_sd_abs": combined.abs().std().values,
    }).sort_values("SHAP_mean_abs", ascending=False).reset_index(drop=True)

    return ShapGroupResult(summary, shap_df, group_shap_df, X_test)

def permutation_importance_with_groups(X_test, y_test, individual_variables, variable_groups, rf, SEED, n_repeats):
    
    rng = np.random.default_rng(SEED)
    baseline_r2 = rf.score(X_test, y_test)
    
    pi_results = []
    
    ################ Individual predictors: shuffle one column at a time
    for col in individual_variables:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            drops.append(baseline_r2 - rf.score(X_perm, y_test))

        pi_results.append({"predictor": col,"PI_mean": np.mean(drops),"PI_sd": np.std(drops, ddof=1)})
    
    
    # Groups: shuffle all columns together using the same row permutation
    for group_name, variables in variable_groups.items():
        cols = [col for col in variables if col in X_test.columns]
        if not cols:
            continue

        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(len(X_test))
            X_perm.loc[:, cols] = X_test[cols].iloc[perm_idx].to_numpy()
            drops.append(baseline_r2 - rf.score(X_perm, y_test))

        pi_results.append({"predictor": group_name,"PI_mean": np.mean(drops),"PI_sd": np.std(drops, ddof=1)})

    pi_df = pd.DataFrame(pi_results).sort_values("PI_mean", ascending=False)

    return pi_df

ShapIndividualResult = namedtuple("ShapIndividualResult", ["summary", "shap_df", "shap_values", "X_shap"])
def shap_values_with_individual_predictors(X_test, rf, SEED, subset_size = None):
    if subset_size is not None:
        X_test = X_test.sample(min(subset_size, len(X_test)), random_state=SEED)
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame(shap_values,columns=X_test.columns,index=X_test.index)

    shap_df_summary = pd.DataFrame({"predictor": X_test.columns,
        "SHAP_mean_abs": shap_df.abs().mean().values}).sort_values("SHAP_mean_abs", ascending=False)
    
    return ShapIndividualResult(shap_df_summary, shap_df, shap_values, X_test)


def permutation_importance_with_individual_predictors(X_test, y_test, rf, SEED):
    pi = permutation_importance(rf, X_test, y_test,scoring="r2", n_repeats=10, random_state=SEED, n_jobs=-1)
    pi_df = pd.DataFrame({"predictor": X_test.columns,"PI_mean": pi.importances_mean,"PI_sd": pi.importances_std,
                         }).sort_values("PI_mean", ascending=False)
    
    return pi_df

# Colours: positive = blue, negative = orange, undefined = grey
def corr_colour(predictor, predictor_corr):
    rho = predictor_corr.get(predictor, np.nan)
    return "steelblue" if rho > 0 else "darkorange" if rho < 0 else "grey"

def find_correlations(subset_early, TARGET, pi_df, grouped_variables):
    predictor_corr = {}
    for predictor in pi_df['predictor']:
        if predictor in grouped_variables:
            cols = [c for c in grouped_variables[predictor] if c in subset_early.columns]
            rhos = [spearmanr(subset_early[c], subset_early[TARGET], nan_policy="omit").statistic for c in cols]
            rho = np.nanmean(rhos) if rhos else np.nan
        else:  # Individual predictor
            if predictor in subset_early.columns:
                rho = spearmanr(subset_early[predictor], subset_early[TARGET], nan_policy="omit").statistic
            else:
                rho = np.nan
        predictor_corr[predictor] = rho
        
    return predictor_corr

def plot_importance_scores(pi_df, shap_df, predictor_corr_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(pi_df) * 0.32)))

    # Permutation importance
    plot_pi = pi_df.sort_values("PI_mean")

    axes[0].barh(plot_pi["predictor"], plot_pi["PI_mean"],xerr=plot_pi["PI_sd"], capsize=2,
                 color=[corr_colour(p, predictor_corr_df) for p in plot_pi["predictor"]])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Drop in held-out R²")
    axes[0].set_title("Permutation importance")

    # Individual SHAP importance
    plot_shap = shap_df.sort_values("SHAP_mean_abs")

    axes[1].barh(plot_shap["predictor"], plot_shap["SHAP_mean_abs"],
        color=[corr_colour(p, predictor_corr_df) for p in plot_shap["predictor"]])
    axes[1].set_xlabel("Mean absolute SHAP value")
    axes[1].set_title("Individual SHAP importance")

    # Shared legend
    legend_handles = [Patch(color="steelblue", label="Positive Spearman correlation"),
                      Patch(color="darkorange", label="Negative Spearman correlation"),
        Patch(color="grey", label="Zero / undefined correlation")]

    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()
    
def groups_hierarchical(corr, thresh, method="average"):
    """
    Cluster variables so that, with the default 'average' linkage, groups
    are merged while mean pairwise |rho| stays >= thresh.
    """
    absc = corr.abs().clip(upper=1.0)
    dist = 1 - absc.values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2          # enforce symmetry for squareform

    Z = linkage(squareform(dist, checks=False), method=method)
    labels = fcluster(Z, t=1 - thresh, criterion="distance")

    groups = {}
    for var, lab in zip(corr.columns, labels):
        groups.setdefault(f"G{lab}", []).append(var)
    return groups

def groups_connected(corr, thresh):
    """
    Connected components of the graph where an edge means |rho| >= thresh.
    Looser than hierarchical: A-B and B-C linked puts A, B, C together
    even if A-C is weak (chaining).
    """
    import networkx as nx
    absc = corr.abs()
    G = nx.Graph()
    G.add_nodes_from(corr.columns)
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if absc.loc[a, b] >= thresh:
                G.add_edge(a, b)
    return {f"G{i+1}": sorted(c) for i, c in enumerate(nx.connected_components(G))}

def group_colour_values(X, cols):
    """One 'group value' per row: mean of z-scored members, sign-aligned to the first member."""
    Z = (X[cols] - X[cols].mean()) / X[cols].std()
    signs = np.sign(Z.corrwith(Z[cols[0]])).replace(0, 1)   # flip members anti-correlated with the first
    return (Z * signs).mean(axis=1)

def make_grouped_explanation(res, variable_groups, individual_variables=None, groups_only=False):
    X = res.X_shap
    parts_shap, parts_data = {}, {}

    if not groups_only and individual_variables is not None:
        for c in individual_variables:
            parts_shap[c] = res.shap_df[c]
            parts_data[c] = X[c]

    for g in res.group_shap_df.columns:
        cols = [c for c in variable_groups[g] if c in X.columns]
        parts_shap[g] = res.group_shap_df[g]
        parts_data[g] = group_colour_values(X, cols)

    shap_part = pd.DataFrame(parts_shap)
    data_part = pd.DataFrame(parts_data)

    return shap.Explanation(
        values=shap_part.values,
        data=data_part.values,
        feature_names=list(shap_part.columns),)    

def plot_shap_maps(grouped, xy, cols=None, ncols=3, pct=98, s=3, cmap="RdBu_r",
                   shared_scale=False, figsize_per_panel=(4.2, 4)):
    """One map per driver, coloured by its SHAP contribution (diverging, centred on 0)."""
    cols = list(cols) if cols is not None else list(grouped.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
                             squeeze=False)

    if shared_scale:
        lim_shared = np.nanpercentile(np.abs(grouped[cols].to_numpy()), pct)

    for ax, col in zip(axes.ravel(), cols):
        v = grouped[col].to_numpy()
        lim = lim_shared if shared_scale else np.nanpercentile(np.abs(v), pct)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=s, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_title(col)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.8, label="SHAP")

    for ax in axes.ravel()[len(cols):]:      # hide unused panels
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Using the outputs of cross-validation
def summarise_grouped_shap(shap_df, individual_variables, variable_groups):
    group_shap = {}
    for group_name, variables in variable_groups.items():
        cols = [c for c in variables if c in shap_df.columns]
        if cols:
            group_shap[group_name] = shap_df[cols].sum(axis=1)
    group_shap_df = pd.DataFrame(group_shap)

    combined = pd.concat([shap_df[individual_variables], group_shap_df], axis=1)
    summary = pd.DataFrame({
        "type": ["individual"] * len(individual_variables) + ["group"] * group_shap_df.shape[1],
        "predictor": combined.columns,
        "SHAP_mean_abs": combined.abs().mean().values,
        "SHAP_sd_abs": combined.abs().std().values,
    }).sort_values("SHAP_mean_abs", ascending=False).reset_index(drop=True)
    return summary, group_shap_df