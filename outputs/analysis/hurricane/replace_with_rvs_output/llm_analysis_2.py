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
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric (coerce non-convertible values to NaN)
    numeric_cols = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep only rows that have the key variables we need for analysis
    # Primary variables: masfem (IV), alldeaths (DV), and intensity controls
    required = ['masfem', 'alldeaths', 'wind', 'category', 'min', 'year']
    present_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=present_required)

    # Standardized femininity score (z-score) for easier interpretation
    df['Femininity_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)

    # Binary female-name indicator (0/1) from gender_mf
    if 'gender_mf' in df.columns:
        # Ensure values are 0/1; coerce anything else to NaN then fill with 0/1 cast
        df['IsFemaleName'] = df['gender_mf'].round().astype('Int64').astype(float).fillna(0).astype(int)
    else:
        df['IsFemaleName'] = 0

    # Dependent variables: log-transform deaths and damage to reduce skew and handle zeros
    df['log_deaths'] = np.log1p(df['alldeaths'].astype(float))
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(pd.to_numeric(df['ndam15'], errors='coerce').fillna(0))
    else:
        df['log_ndam15'] = np.nan

    # Keep only columns needed for modeling plus original identifiers for traceability
    keep_cols = ['ind', 'name'] if 'ind' in df.columns and 'name' in df.columns else []
    keep_cols += ['masfem', 'Femininity_z', 'gender_mf', 'IsFemaleName', 'alldeaths', 'log_deaths', 'ndam15', 'log_ndam15', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols]

    # Final dropna for model columns to ensure model functions get complete rows
    # (model function will also perform its own dropna to be safe)
    model_req = ['Femininity_z', 'log_deaths', 'wind', 'category', 'min', 'year']
    model_req = [c for c in model_req if c in df.columns]
    df = df.dropna(subset=model_req)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # This function runs the primary and a robustness model.
    # Primary test: does name femininity predict fatalities after controlling for objective storm severity?

    # Prepare model variables
    X_cols = ['Femininity_z', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'IsFemaleName']
    X_cols = [c for c in X_cols if c in df.columns]

    # Ensure there are no missing values in the model matrix for the primary model
    df_model = df.dropna(subset=X_cols + ['log_deaths'])

    # If there are too few rows, raise a warning by returning an empty result dictionary
    if df_model.shape[0] < 10:
        return {'error': 'Not enough rows after cleaning to fit models', 'n_rows': int(df_model.shape[0])}

    # Build design matrix for fatalities model
    X = sm.add_constant(df_model[X_cols])
    y = df_model['log_deaths']

    # OLS with robust standard errors (HC3) to reduce sensitivity to heteroskedasticity
    deaths_model = sm.OLS(y, X).fit(cov_type='HC3')

    # Robustness: run the same specification predicting monetary damage (if present)
    damage_results = None
    if 'log_ndam15' in df.columns:
        df_damage = df.dropna(subset=X_cols + ['log_ndam15'])
        if df_damage.shape[0] >= 10:
            X2 = sm.add_constant(df_damage[X_cols])
            y2 = df_damage['log_ndam15']
            damage_model = sm.OLS(y2, X2).fit(cov_type='HC3')
            damage_results = damage_model

    # Return fitted model results objects so the caller can inspect summary(), params, etc.
    return {
        'n_obs_deaths_model': int(df_model.shape[0]),
        'deaths_model': deaths_model,
        'n_obs_damage_model': (int(df_damage.shape[0]) if 'df_damage' in locals() else 0),
        'damage_model': damage_results
    }


