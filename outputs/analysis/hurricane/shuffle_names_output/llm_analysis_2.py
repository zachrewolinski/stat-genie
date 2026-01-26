from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Required raw columns we expect in this dataset
    # 'name' = coder-rated masculinity<->femininity index (higher => more feminine)
    # 'ndam15' = total deaths caused by the hurricane
    # 'ind' = normalized property damage (raw dollar amounts adjusted)
    # 'wind' = maximum wind speed at landfall
    # 'min' = minimum central pressure at landfall
    # 'masfem' = Saffir-Simpson category / numeric intensity code
    # 'alldeaths' = year of storm (dataset naming is idiosyncratic)
    # 'elapsedyrs' = binary gender indicator for the name (0 male, 1 female)

    # Drop rows missing the minimal set needed for the main analysis
    required = [c for c in ['name', 'ndam15', 'wind', 'min', 'masfem', 'alldeaths'] if c in df.columns]
    if len(required) > 0:
        df = df.dropna(subset=required)

    # Winsorize deaths and damage at 1st and 99th percentiles to reduce extreme influence
    if 'ndam15' in df.columns:
        low = df['ndam15'].quantile(0.01)
        high = df['ndam15'].quantile(0.99)
        df['ndam15_winsor'] = df['ndam15'].clip(lower=low, upper=high)
        # log transform (log1p to handle zeros)
        df['log_deaths'] = np.log1p(df['ndam15_winsor'])

    if 'ind' in df.columns:
        low = df['ind'].quantile(0.01)
        high = df['ind'].quantile(0.99)
        df['ind_winsor'] = df['ind'].clip(lower=low, upper=high)
        df['log_damage'] = np.log1p(df['ind_winsor'])

    # Standardize (z-score) the continuous predictors/controls used in the regression
    # Note: use population sd (ddof=0) to be explicit and stable for small samples
    if 'name' in df.columns:
        df['femininity_z'] = (df['name'] - df['name'].mean()) / (df['name'].std(ddof=0) if df['name'].std(ddof=0) != 0 else 1.0)

    if 'wind' in df.columns:
        df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1.0)

    if 'min' in df.columns:
        df['min_z'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1.0)

    if 'masfem' in df.columns:
        df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # 'alldeaths' in this dataset actually encodes the year of the storm
    if 'alldeaths' in df.columns:
        df['year_z'] = (df['alldeaths'] - df['alldeaths'].mean()) / (df['alldeaths'].std(ddof=0) if df['alldeaths'].std(ddof=0) != 0 else 1.0)

    # Binary indicator if present
    if 'elapsedyrs' in df.columns:
        # ensure it's 0/1 integer
        df['is_female_name'] = df['elapsedyrs'].astype(int)

    # Drop any rows missing our primary IV or DV
    required_for_model = [c for c in ['femininity_z', 'log_deaths'] if c in df.columns]
    if len(required_for_model) > 0:
        df = df.dropna(subset=required_for_model)

    # Keep only columns relevant for modeling (plus raw columns for reference)
    keep_cols = [
        'femininity_z', 'log_deaths', 'log_damage',
        'wind_z', 'min_z', 'masfem_z', 'year_z', 'is_female_name',
        # keep raw originals for traceability if present
        'name', 'ndam15', 'ind', 'wind', 'min', 'masfem', 'alldeaths', 'elapsedyrs'
    ]
    # Filter to columns that exist in df
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    df = df.copy()

    # Build main model predicting log fatalities
    # Select controls that exist in the transformed dataframe
    base_controls = [c for c in ['wind_z', 'min_z', 'masfem_z', 'year_z', 'is_female_name'] if c in df.columns]

    # Prepare design matrix for main model
    X_cols = ['femininity_z'] + base_controls
    X = df[X_cols].astype(float)
    X = sm.add_constant(X)
    y = df['log_deaths'].astype(float)

    # Fit OLS with robust (HC3) standard errors
    model_main = sm.OLS(y, X).fit(cov_type='HC3')

    # Interaction model: femininity * wind (tests whether effect of perceived femininity differs by intensity)
    interaction_results = None
    if 'wind_z' in df.columns:
        df['fem_x_wind'] = df['femininity_z'] * df['wind_z']
        X_int_cols = X_cols + ['fem_x_wind']
        X_int = df[X_int_cols].astype(float)
        X_int = sm.add_constant(X_int)
        interaction_results = sm.OLS(y, X_int).fit(cov_type='HC3')

    # Robustness model: predict log property damage (alternative outcome)
    damage_results = None
    if 'log_damage' in df.columns:
        y2 = df['log_damage'].astype(float)
        X2 = df[X_cols].astype(float)
        X2 = sm.add_constant(X2)
        damage_results = sm.OLS(y2, X2).fit(cov_type='HC3')

    # Return fitted model objects (statsmodels RegressionResults). Caller can print .summary() or extract coefficients.
    return {
        'deaths_model': model_main,
        'interaction_model': interaction_results,
        'damage_model': damage_results
    }


