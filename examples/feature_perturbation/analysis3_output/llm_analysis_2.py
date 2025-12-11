from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Standardize column names that may be confused in metadata; keep exact final column names used in modeling
    # Columns expected in raw data: 'name' (masc-fem index), 'ndam15' (total deaths), 'wind', 'min', 'ind', 'alldeaths', 'masfem', 'masfem_mturk'

    # Ensure numeric types (coerce non-numeric to NaN)
    numeric_cols = ['name', 'ndam15', 'wind', 'min', 'ind', 'alldeaths', 'masfem', 'masfem_mturk']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core IV or DV (we cannot model without them)
    df = df.dropna(subset=['name', 'ndam15'])

    # For controls, we will allow some missingness but drop rows missing the primary set of controls
    # (wind, min, ind, alldeaths). If you want to keep more rows, consider multiple imputation.
    controls_needed = [c for c in ['wind', 'min', 'ind', 'alldeaths'] if c in df.columns]
    if len(controls_needed) > 0:
        df = df.dropna(subset=controls_needed)

    # Create a log-transformed deaths variable for OLS sensitivity analysis
    df['log_ndam15'] = np.log1p(df['ndam15'])

    # Optionally standardize the IV for interpretable coefficients in sensitivity checks
    df['name_z'] = (df['name'] - df['name'].mean()) / (df['name'].std(ddof=0) if df['name'].std(ddof=0) != 0 else 1)

    # Trim or winsorize extremely large damage values to reduce influence of outliers (robustness)
    if 'ind' in df.columns:
        # Replace negative or zero with small positive to allow log transforms if needed later
        df['ind'] = df['ind'].apply(lambda x: np.nan if pd.isna(x) else (x if x >= 0 else np.nan))
        # create log damage variable
        df['log_ind'] = np.log1p(df['ind'])

    # Ensure final dataframe contains exactly the columns described in conceptual variables + auxiliaries
    final_cols = [c for c in ['name', 'name_z', 'ndam15', 'log_ndam15', 'wind', 'min', 'ind', 'log_ind', 'alldeaths', 'masfem', 'masfem_mturk'] if c in df.columns]
    df = df[final_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs primary negative-binomial GLM on death counts and an OLS on log(deaths+1) as sensitivity.

    Returns a dict with both fitted results objects (statsmodels result instances).
    """
    results = {}

    # Prepare design matrix for controls used in primary specification
    controls = [c for c in ['wind', 'min', 'ind', 'alldeaths', 'masfem'] if c in df.columns]

    # Primary model: negative binomial regression for count of deaths (handles overdispersion)
    try:
        exog_vars = ['name'] + controls
        X = df[exog_vars]
        X = sm.add_constant(X, has_constant='add')
        y = df['ndam15']

        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
        results['nb_model'] = nb_model
    except Exception as e:
        results['nb_model_error'] = str(e)

    # Sensitivity: linear regression on log( ndam15 + 1 )
    try:
        exog_vars = ['name'] + controls
        X2 = df[exog_vars]
        X2 = sm.add_constant(X2, has_constant='add')
        y2 = df['log_ndam15']

        ols_model = sm.OLS(y2, X2).fit()
        results['ols_model'] = ols_model
    except Exception as e:
        results['ols_model_error'] = str(e)

    # Additional robustness check: use standardized IV (name_z) in OLS if present
    if 'name_z' in df.columns:
        try:
            exog_vars_z = ['name_z'] + controls
            X3 = df[exog_vars_z]
            X3 = sm.add_constant(X3, has_constant='add')
            ols_z = sm.OLS(df['log_ndam15'], X3).fit()
            results['ols_name_z'] = ols_z
        except Exception as e:
            results['ols_name_z_error'] = str(e)

    return results


