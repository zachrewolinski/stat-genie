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
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric (coerce errors to NaN)
    num_cols = ['masfem', 'masfem_mturk', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create log-transformed dependent variables (log(1 + x)) to handle zeros and skew
    if 'alldeaths' in df.columns:
        df['log_alldeaths'] = np.log1p(df['alldeaths'])
    else:
        df['log_alldeaths'] = np.nan

    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        df['log_ndam15'] = np.nan

    # Standardize the continuous femininity measures (z-scores). Use sample mean/std ignoring NaNs.
    if 'masfem' in df.columns:
        mean_m = df['masfem'].mean()
        std_m = df['masfem'].std(ddof=0)
        df['masfem_z'] = (df['masfem'] - mean_m) / (std_m if std_m != 0 else 1.0)
    else:
        df['masfem_z'] = np.nan

    if 'masfem_mturk' in df.columns:
        mean_mt = df['masfem_mturk'].mean()
        std_mt = df['masfem_mturk'].std(ddof=0)
        df['masfem_mturk_z'] = (df['masfem_mturk'] - mean_mt) / (std_mt if std_mt != 0 else 1.0)
    else:
        df['masfem_mturk_z'] = np.nan

    # Binary female name indicator (ensure numeric 0/1). Keep NaN if missing.
    if 'gender_mf' in df.columns:
        df['female_name'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype(float)
    else:
        df['female_name'] = np.nan

    # Year centered to improve interpretability and numerical stability
    if 'year' in df.columns:
        df['year_c'] = df['year'] - df['year'].median()
    else:
        df['year_c'] = np.nan

    # Keep source as-is (categorical string) to be handled by the modeling function via C(source)

    # Drop rows that are missing the core independent variables or core controls (these rows cannot be used in the primary regression)
    required_for_model = ['masfem_z', 'wind', 'category', 'min', 'year_c', 'elapsedyrs', 'source']
    present_required = [c for c in required_for_model if c in df.columns]
    if len(present_required) > 0:
        df = df.dropna(subset=present_required)

    # Return transformed dataframe with all columns that will be used by the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # We'll run: (1) primary OLS predicting log fatalities from name femininity (masfem_z)
    # with controls for physical storm severity and data source. We use robust (HC3) SEs.
    # (2) Robustness: replace masfem_z with binary female_name
    # (3) Secondary outcome: property damage (log_ndam15) with the same covariates.

    import statsmodels.formula.api as smf

    results = {}

    # Primary model: fatalities
    formula_primary = 'log_alldeaths ~ masfem_z + wind + category + min + year_c + elapsedyrs + C(source)'
    try:
        res_primary = smf.ols(formula_primary, data=df).fit(cov_type='HC3')
        results['alldeaths_masfem'] = res_primary
    except Exception as e:
        results['alldeaths_masfem'] = e

    # Robustness: binary female name instead of continuous masfem
    formula_gender = 'log_alldeaths ~ female_name + wind + category + min + year_c + elapsedyrs + C(source)'
    try:
        res_gender = smf.ols(formula_gender, data=df).fit(cov_type='HC3')
        results['alldeaths_femaleName'] = res_gender
    except Exception as e:
        results['alldeaths_femaleName'] = e

    # Secondary outcome: property damage (log_ndam15) as a robustness check / alternative proxy
    formula_damage = 'log_ndam15 ~ masfem_z + wind + category + min + year_c + elapsedyrs + C(source)'
    try:
        res_damage = smf.ols(formula_damage, data=df).fit(cov_type='HC3')
        results['ndam15_masfem'] = res_damage
    except Exception as e:
        results['ndam15_masfem'] = e

    # Return a dict of fitted models (or exceptions if any model could not be fit).
    # The caller can inspect .summary() on each statsmodels RegressionResults object.
    return results


