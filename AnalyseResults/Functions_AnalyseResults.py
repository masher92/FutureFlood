from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def partial_r2_for_catchment(df_catch, flood_metric='10cm_area'):
    """
    Returns incremental R² of soil moisture features beyond rainfall + storm shape.
    """
    # Drop rows with nulls in relevant columns
    sm_cols = ['mean_sm_2_before_event']
    
    base_features = ['max_precip', 'gini', 'd50', 'peak_position_ratio',
                     'dry_ratio', 'time_based_std', 'total_acc']
    
    all_cols = base_features + sm_cols + [flood_metric]
    df = df_catch[all_cols].dropna()
    
    if len(df) < 20:  # skip catchments with too few events
        return np.nan
    
    X_base = df[base_features].values
    X_full = df[base_features + sm_cols].values
    y = df[flood_metric].values
    
    # Standardise
    scaler_base = StandardScaler()
    scaler_full = StandardScaler()
    X_base_s = scaler_base.fit_transform(X_base)
    X_full_s = scaler_full.fit_transform(X_full)
    
    r2_base = r2_score(y, LinearRegression().fit(X_base_s, y).predict(X_base_s))
    r2_full = r2_score(y, LinearRegression().fit(X_full_s, y).predict(X_full_s))
    
    return max(r2_full - r2_base, 0)  # clip at 0; negative = SM adds nothing


def plot_with_best_fit_line(results_df, ax, col1, col2, label1, label2, title):
    x = results_df[col1]
    y = results_df[col2]

    ax.scatter(x, y)

    # 🔹 Fit line
    # m, b = np.polyfit(x, y, 1)

    # Predicted values
    y_pred = m * x + b

    # R²
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)

    # 🔹 Create line values
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b

    # 🔹 Plot line
    ax.plot(x_line, y_line, color='black')

    ax.text( 0.05, 0.95,  f'$R^2 = {r2:.2f}$', transform=ax.transAxes, verticalalignment='top')

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    
    ax.set_title(title)
