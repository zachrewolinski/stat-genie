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
    Produce a dataframe containing all columns necessary for modeling the relationship between
    hurricane name femininity and mortality/damage outcomes. The function:
      - ensures numeric types for key columns
      - drops rows missing essential variables for the main analysis
      - creates log-transformed outcome variables
      - standardizes femininity ratings (masfem and masfem_mturk)

    Resulting dataframe contains at minimum the columns:
      ['alldeaths','ndam15','masfem','masfem_mturk','gender_mf','wind','category','min','elapsedyrs','year','source',
       'log_alldeaths','log_ndam15','masfem_std','masfem_mturk_std']
    """
    df = df.copy()

    # Ensure numeric types where appropriate
    numeric_cols = ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure source exists as string/categorical
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Drop rows missing the primary dependent and independent variables and core controls
    required_for_main = ['alldeaths', 'masfem', 'wind', 'category', 'min', 'elapsedyrs', 'source']
    missing_required = [c for c in required_for_main if c not in df.columns]
    if missing_required:
        raise ValueError(f"Input dataframe is missing required columns for the main analysis: {missing_required}")

    df = df.dropna(subset=required_for_main)

    # Create log-transformed outcomes (add-one to handle zeros)
    df['log_alldeaths'] = np.log1p(df['alldeaths'])
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        df['log_ndam15'] = np.nan

    # Standardize masfem (and masfem_mturk if present)
    df['masfem_std'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_std'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
    else:
        df['masfem_mturk_std'] = np.nan

    # Ensure binary gender variable is integer (0/1)
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].astype('Int64')

    # Keep columns necessary for modeling to avoid accidental use of other columns
    keep_cols = ['alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15', 'masfem', 'masfem_std', 'masfem_mturk', 'masfem_mturk_std', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year', 'source']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit regression models to test whether more feminine hurricane names are associated with
    higher fatalities (as a proxy for lower precautionary responses). We estimate:
      1) OLS on log(alldeaths + 1) with masfem (standardized) and controls; robust SEs
      2) OLS on log(alldeaths + 1) with gender_mf (binary) and same controls (robust SEs)
      3) Robustness: OLS on log(ndam15 + 1) with masfem_std and controls

    Returns a dictionary of fitted results objects.
    """
    import statsmodels.formula.api as smf
    results = {}

    # Basic checks
    for col in ['log_alldeaths', 'masfem_std', 'wind', 'category', 'min', 'elapsedyrs', 'source']:
        if col not in df.columns:
            raise ValueError(f"Column required for modeling not found in dataframe: {col}")

    # Model 1: continuous femininity measure
    formula1 = 'log_alldeaths ~ masfem_std + wind + category + min + elapsedyrs + C(source)'
    ols_masfem = smf.ols(formula1, data=df).fit(cov_type='HC3')
    results['ols_masfem'] = ols_masfem

    # Model 2: binary gender indicator
    if 'gender_mf' in df.columns and df['gender_mf'].notnull().any():
        formula2 = 'log_alldeaths ~ gender_mf + wind + category + min + elapsedyrs + C(source)'
        ols_gender = smf.ols(formula2, data=df).fit(cov_type='HC3')
        results['ols_gender_binary'] = ols_gender

    # Model 3: robustness using inflation-adjusted damage
    if 'log_ndam15' in df.columns and df['log_ndam15'].notnull().any():
        formula3 = 'log_ndam15 ~ masfem_std + wind + category + min + elapsedyrs + C(source)'
        ols_damage = smf.ols(formula3, data=df).fit(cov_type='HC3')
        results['ols_damage_masfem'] = ols_damage

    # Return the fitted results objects; the caller can inspect .summary() for each
    return results


