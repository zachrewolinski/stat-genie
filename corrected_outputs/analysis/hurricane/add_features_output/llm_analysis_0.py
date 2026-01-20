from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the analysis dataframe.

    Produces standardized IVs and controls and log-transformed DV for OLS.
    Drops rows with missing values on variables needed for the main analyses.
    """
    df = df.copy()

    # Ensure numeric for key columns (coerce errors -> NaN)
    num_cols = ['masfem', 'gender_mf', 'wind', 'category', 'min', 'alldeaths', 'ndam15', 'year', 'masfem_mturk', 'elapsedyrs']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep rows with the data needed for main analysis
    required = ['masfem', 'gender_mf', 'wind', 'category', 'min', 'alldeaths', 'year']
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError('Missing required columns in input dataframe: {}'.format(missing_required))

    df = df.dropna(subset=required)

    # Dependent variable transformations
    # Raw counts kept for count models; log transform for OLS
    df['alldeaths'] = df['alldeaths'].astype(float)
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # If economic damage is of interest, also prepare log damage (robustness)
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log_ndam15'] = np.log(df['ndam15'].fillna(0) + 1)

    # Binary female indicator (ensure 0/1)
    df['gender_female'] = df['gender_mf'].astype(int)

    # Standardize continuous predictors and controls (z-scores). Use population std (ddof=0) for stability.
    # Standardize both masfem measures if available
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_mturk_z'] = np.nan

    # Standardize objective storm severity controls
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1)
    df['min_z'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1)
    df['category_z'] = (df['category'] - df['category'].mean()) / (df['category'].std(ddof=0) if df['category'].std(ddof=0) != 0 else 1)
    df['year_z'] = (df['year'] - df['year'].mean()) / (df['year'].std(ddof=0) if df['year'].std(ddof=0) != 0 else 1)

    # Final check: drop any remaining rows with missing values in model columns
    model_cols = ['masfem_z', 'gender_female', 'wind_z', 'min_z', 'category_z', 'year_z', 'alldeaths', 'log_alldeaths']
    df = df.dropna(subset=model_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run primary statistical models to test whether more feminine hurricane names are associated
    with larger human impact (proxy for fewer precautions) after controlling for objective storm severity.

    Returns a dictionary with:
      - 'ols': OLS fit on log(alldeaths + 1) with robust standard errors
      - 'nb': Negative binomial GLM on alldeaths (counts)
      - 'ols_mturk': OLS robustness using masfem_mturk_z in place of masfem_z (if available)

    The models include controls: wind_z, min_z, category_z, year_z. The independent variables are
    masfem_z (primary continuous femininity measure) and gender_female (binary female name indicator).
    """
    df = df.copy()

    # Prepare design matrix columns
    base_covariates = ['masfem_z', 'gender_female', 'wind_z', 'min_z', 'category_z', 'year_z']
    missing = [c for c in base_covariates if c not in df.columns]
    if missing:
        raise ValueError('Missing columns required for modeling: {}'.format(missing))

    X = df[base_covariates]
    X = sm.add_constant(X)

    # Outcome for OLS (log-transformed deaths)
    y_ols = df['log_alldeaths']
    ols_res = sm.OLS(y_ols, X).fit(cov_type='HC3')

    # Count model: Negative Binomial on raw death counts
    # Use GLM with NegativeBinomial family to allow overdispersion relative to Poisson
    X_nb = X.copy()
    y_nb = df['alldeaths'].astype(float)
    try:
        nb_res = sm.GLM(y_nb, X_nb, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback to Poisson if NegativeBinomial fails to converge
        nb_res = sm.GLM(y_nb, X_nb, family=sm.families.Poisson()).fit()

    # Robustness: use masfem_mturk_z (alternative femininity measure) if available
    if 'masfem_mturk_z' in df.columns and df['masfem_mturk_z'].notnull().any():
        X_rob = X.copy()
        X_rob['masfem_z'] = df['masfem_mturk_z']
        ols_rob = sm.OLS(y_ols, X_rob).fit(cov_type='HC3')
    else:
        ols_rob = None

    results = {
        'ols': ols_res,
        'nb': nb_res,
        'ols_mturk': ols_rob
    }

    return results


