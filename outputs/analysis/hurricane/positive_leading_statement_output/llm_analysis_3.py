from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/positive_leading_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into a modeling-ready dataframe.

    Produces the following required columns (exact names used in modeling):
      - alldeaths (int / numeric)
      - ndam15 (numeric original)
      - log_ndam15 (np.log(ndam15 + 1))
      - masfem_z (standardized femininity rating)
      - masfem_mturk_z (standardized MTurk femininity rating, if available)
      - gender_mf (binary: 0 male, 1 female)
      - wind, category, min, year, elapsedyrs
      - source_uri, source_wiki, source_mwr (binary indicators of source)

    Rows with missing key variables are dropped.
    """
    df = df.copy()

    # Ensure numeric columns are numeric; coerce errors to NaN
    num_cols = ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'gender_mf', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep only rows with required columns for the principal analyses
    required_for_primary = ['alldeaths', 'ndam15', 'masfem', 'wind', 'category', 'min', 'year']
    for c in required_for_primary:
        if c not in df.columns:
            raise KeyError(f"Required column not found in input dataframe: {c}")

    df = df.dropna(subset=required_for_primary)

    # Create log-transformed damage outcome
    df['log_ndam15'] = np.log(df['ndam15'] + 1)

    # Standardize masfem and masfem_mturk for interpretability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        # Create column of NaNs so it exists for model specification / robustness checks
        df['masfem_mturk_z'] = np.nan

    # Clean source and create deterministic binary flags so model code can reference exact names
    df['source_clean'] = df['source'].astype(str).str.lower()
    df['source_uri'] = df['source_clean'].str.contains('uri', na=False).astype(int)
    df['source_wiki'] = df['source_clean'].str.contains('wiki', na=False).astype(int)
    df['source_mwr'] = df['source_clean'].str.contains('mwr', na=False).astype(int)

    # Ensure gender_mf is binary 0/1 if present; if missing, fill with 0/NaN
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].fillna(0).astype(int)
    else:
        df['gender_mf'] = 0

    # Keep only columns needed for modeling plus a few extras for diagnostics
    keep_cols = [
        'alldeaths', 'ndam15', 'log_ndam15', 'masfem', 'masfem_z', 'masfem_mturk_z', 'gender_mf',
        'wind', 'category', 'min', 'year', 'elapsedyrs', 'source_uri', 'source_wiki', 'source_mwr', 'name'
    ]
    # Keep intersection of available columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to test whether more feminine hurricane names are associated
    with worse outcomes (consistent with fewer precautionary measures):
      1) Negative binomial regression for alldeaths (count outcome).
      2) OLS regression for logged property damage (log_ndam15).

    Both models adjust for storm intensity (wind, category, min pressure), year, name-gender
    and data source indicators. Returns fitted result objects (robust covariance estimates).
    """
    # Required modeling columns
    model_cols = ['masfem_z', 'wind', 'category', 'min', 'year', 'gender_mf', 'source_uri', 'source_wiki', 'source_mwr']
    for c in model_cols:
        if c not in df.columns:
            raise KeyError(f"Required modeling column missing from transformed df: {c}")

    # Prepare design matrix
    X = df[model_cols].copy()
    X = sm.add_constant(X)

    results = {}

    # 1) Negative Binomial for fatalities (alldeaths)
    y_deaths = df['alldeaths']
    # Fit GLM negative binomial; then compute robust covariance (HC3)
    nb_glm = sm.GLM(y_deaths, X, family=sm.families.NegativeBinomial())
    nb_res_raw = nb_glm.fit()
    try:
        nb_res = nb_res_raw.get_robustcov_results(cov_type='HC3')
    except Exception:
        # fallback to the raw result if robustcov is not available
        nb_res = nb_res_raw
    print('\nNegative Binomial (alldeaths) results:')
    print(nb_res.summary())
    results['deaths_nb'] = nb_res

    # 2) OLS for logged damage (log_ndam15)
    if 'log_ndam15' in df.columns:
        y_damage = df['log_ndam15']
        ols = sm.OLS(y_damage, X)
        ols_res_raw = ols.fit()
        try:
            ols_res = ols_res_raw.get_robustcov_results(cov_type='HC3')
        except Exception:
            ols_res = ols_res_raw
        print('\nOLS (log damage) results:')
        print(ols_res.summary())
        results['damage_ols'] = ols_res
    else:
        results['damage_ols'] = None

    # Return the fitted results for downstream inspection / tests
    return results


