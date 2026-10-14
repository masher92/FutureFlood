import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, cross_validate

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

def grouped_permutation_importance(model, X, y, groups, n_repeats=10, random_state=42):
    """
    groups: dict mapping group_name -> list of column names to permute together
    """
    rng = np.random.RandomState(random_state)
    baseline_score = model.score(X, y)
    results = {}
    
    for group_name, cols in groups.items():
        drop_scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            # Shuffle all columns in the group using the SAME row order,
            # so their joint relationship is scrambled together
            shuffled_idx = rng.permutation(len(X_permuted))
            X_permuted[cols] = X_permuted[cols].values[shuffled_idx]
            score = model.score(X_permuted, y)
            drop_scores.append(baseline_score - score)
        results[group_name] = {
            'importance_mean': np.mean(drop_scores),
            'importance_std': np.std(drop_scores)
        }
    return pd.DataFrame(results).T.sort_values('importance_mean', ascending=False)