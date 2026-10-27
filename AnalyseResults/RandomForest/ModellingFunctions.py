from scipy.stats import zscore, linregress
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import numpy as np

def find_linear_relationship (df, variable1):
    x = df[variable1]
    y = df["10cm_area"]

    # Simple regression: flood area ~ rainfall intensity
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r2_simple = r_value**2
    if p_value <0.005:
        p_value = '<0.05'
    else:
        p_value = round(p_value, 2)
    # print(" Simple:", f"{r2_simple:.2f}", p_value)  
    return [r2_simple, p_value]


def groups_hierarchical(corr, thresh=0.75, method="average"):
    """Cluster on distance = 1 - |rho|. 'average' needs members to be correlated with each other
    on average (compact groups); 'single' lets chains form (A~B, B~C => A,B,C); 'complete' is strictest
    (every pair in a group must exceed thresh)."""
    dist = (1 - corr.abs().fillna(0)).values.copy()
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method=method)
    labels = fcluster(Z, t=1 - thresh, criterion="distance")
    return _labels_to_groups(corr.columns, labels)


def groups_connected(corr, thresh=0.75):
    """Link any pair with |rho| >= thresh; each connected component is a group (chaining allowed)."""
    adj = (corr.abs().fillna(0).values >= thresh)
    n, labels = connected_components(adj, directed=False)
    return _labels_to_groups(corr.columns, labels)


def _labels_to_groups(cols, labels):
    out = {}
    for lab in np.unique(labels):
        members = list(cols[labels == lab])
        if len(members) > 1:                      # only report multi-variable groups
            out[f"group_{'_'.join(members[0].split('_')[:2])}"] = members
    return out

