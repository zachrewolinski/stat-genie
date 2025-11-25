from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into a modeling-ready dataframe.

    Produces the following new columns used in the model:
    - log_alldeaths: np.log1p(alldeaths)
    - log_ndam15: np.log1p(ndam15)
    - masfem_z: standardized masfem (z-score)
    - masfem_mturk_z: standardized masfem_mturk (z-score) kept for robustness/inspection
    - source_code: categorical integer code for 'source'

    Also drops rows with missing values in the variables required for modeling.
    """
    df = df.copy()

    # Keep only columns needed for transformations and modelling
    needed = [
        'alldeaths', 'masfem', 'masfem_mturk', 'wind', 'category', 'min',
        'elapsedyrs', 'ndam15', 'source', 'gender_mf'
    ]

    # If any needed columns are missing from the incoming df, raise a clearer error
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Input dataframe is missing required columns: {missing_cols}")

    df = df[needed].copy()

    # Drop rows with missing values in the core variables we'll use
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'category', 'min', 'elapsedyrs', 'ndam15', 'source'])

    # Dependent variable: log transform of fatalities to reduce skew and handle zeros
    df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Log transform of damages (2015-adjusted) as a control
    df['log_ndam15'] = np.log1p(df['ndam15'])

    # Standardize masfem ratings (z-score) for interpretable coefficient
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Also standardize masfem_mturk for robustness checks
    if df['masfem_mturk'].notna().any():
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
    else:
        df['masfem_mturk_z'] = np.nan

    # Encode source as a categorical integer code to control for source differences
    df['source_code'] = df['source'].astype('category').cat.codes

    # Ensure gender_mf is numeric (0/1). If it's not numeric, attempt conversion
    if df['gender_mf'].dtype == 'O':
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')

    # Final drop of any remaining rows with NA in columns used by the model
    model_vars = ['log_alldeaths', 'masfem_z', 'wind', 'category', 'min', 'elapsedyrs', 'log_ndam15', 'gender_mf', 'source_code']
    df = df.dropna(subset=model_vars)

    # Reset index for convenience
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model predicting log(alldeaths + 1) from masfem_z controlling for storm intensity
    and other covariates. Uses robust (heteroskedasticity-consistent) standard errors.

    Returns the fitted statsmodels regression results object.
    """
    # Expect df to already be transformed by transform()
    # Verify required columns exist
    required = ['log_alldeaths', 'masfem_z', 'wind', 'category', 'min', 'elapsedyrs', 'log_ndam15', 'gender_mf', 'source_code']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Specify regressors
    control_cols = ['wind', 'category', 'min', 'elapsedyrs', 'log_ndam15', 'gender_mf', 'source_code']
    X_cols = ['masfem_z'] + control_cols

    X = df[X_cols].astype(float)
    X = sm.add_constant(X)
    y = df['log_alldeaths'].astype(float)

    # Fit OLS with robust (HC3) standard errors
    model_res = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted results object so the caller can inspect coefficients, p-values, etc.
    return model_res


