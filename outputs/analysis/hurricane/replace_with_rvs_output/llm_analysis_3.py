from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying input
    df = df.copy()

    # Ensure numeric columns have the right dtype where possible
    numeric_cols = ['masfem', 'masfem_mturk', 'gender_mf', 'wind', 'category', 'min', 'alldeaths', 'ndam15', 'elapsedyrs', 'year']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the primary variables needed for the main analysis
    required = ['masfem', 'ndam15', 'wind', 'category', 'min', 'year']
    df = df.dropna(subset=required)

    # Create log-transformed dependent variables (add 1 to handle zeros)
    df['log_ndam15'] = np.log(df['ndam15'] + 1)
    df['log_alldeaths'] = np.log(df['alldeaths'].fillna(0) + 1) if 'alldeaths' in df.columns else np.nan

    # Standardize femininity measures (z-scores). Use population std (ddof=0) for reproducibility.
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_mturk_z'] = np.nan

    # Ensure gender_mf is integer 0/1
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].astype(pd.Int64Dtype()).fillna(0).astype(int)
    else:
        df['gender_mf'] = 0

    # Center year to aid interpretation and numerical stability
    df['year_center'] = df['year'] - df['year'].mean()

    # Ensure elapsedyrs exists; if missing, create from year and a reference year if possible
    if 'elapsedyrs' not in df.columns or df['elapsedyrs'].isna().all():
        # If elapsedyrs not provided, create as (max year - year) as a fallback
        df['elapsedyrs'] = df['year'].max() - df['year']
    else:
        df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce').fillna(df['year'].max() - df['year'])

    # Encode source as a single categorical code to control for source differences
    if 'source' in df.columns:
        df['source_code'] = pd.Categorical(df['source']).codes
    else:
        df['source_code'] = 0

    # Keep only columns that will be used in modeling plus original identifiers
    keep_cols = ['ind', 'year', 'year_center', 'masfem', 'masfem_z', 'masfem_mturk_z', 'gender_mf', 'wind', 'category', 'min', 'ndam15', 'log_ndam15', 'alldeaths', 'log_alldeaths', 'elapsedyrs', 'source_code']
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # Build the design matrix for the primary model
    df = df.copy()

    # Define predictors: primary IV (masfem_z) plus controls. Exclude rows with missing outcome or main IV.
    model_cols = ['masfem_z', 'gender_mf', 'wind', 'category', 'min', 'year_center', 'elapsedyrs', 'source_code']
    model_cols = [c for c in model_cols if c in df.columns]

    # Drop rows with missing predictors or outcome
    df_model = df.dropna(subset=['log_ndam15'] + model_cols)

    X = df_model[model_cols]
    X = sm.add_constant(X)
    y = df_model['log_ndam15']

    # Primary OLS with robust (HC3) standard errors
    ols_model = sm.OLS(y, X).fit(cov_type='HC3')

    # Sensitivity 1: use masfem_mturk_z instead of masfem_z (if available)
    sensitivity = None
    if 'masfem_mturk_z' in df.columns and df['masfem_mturk_z'].notna().any():
        sens_cols = ['masfem_mturk_z', 'gender_mf', 'wind', 'category', 'min', 'year_center', 'elapsedyrs', 'source_code']
        sens_cols = [c for c in sens_cols if c in df.columns]
        df_sens = df.dropna(subset=['log_ndam15'] + sens_cols)
        if len(df_sens) > 0:
            Xs = sm.add_constant(df_sens[sens_cols])
            ys = df_sens['log_ndam15']
            sensitivity = sm.OLS(ys, Xs).fit(cov_type='HC3')

    # Sensitivity 2: outcome = log(alldeaths + 1) (fatalities) to test the same directional hypothesis
    deaths_model = None
    if 'log_alldeaths' in df.columns and df['log_alldeaths'].notna().any():
        death_cols = [c for c in model_cols if c in df.columns]
        df_death = df.dropna(subset=['log_alldeaths'] + death_cols)
        if len(df_death) > 0:
            Xd = sm.add_constant(df_death[death_cols])
            yd = df_death['log_alldeaths']
            deaths_model = sm.OLS(yd, Xd).fit(cov_type='HC3')

    # Return a dictionary of fitted models (statsmodels result objects). Caller can inspect .summary().
    return {
        'primary_model': ols_model,
        'sensitivity_mturk_model': sensitivity,
        'deaths_model': deaths_model
    }


