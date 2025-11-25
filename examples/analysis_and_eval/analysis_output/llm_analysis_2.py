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
    Transform the raw hurricane dataframe into the analysis-ready dataframe.

    Produces:
    - log_alldeaths: np.log1p(alldeaths)
    - log_ndam15: np.log1p(ndam15)
    - masfem_z: standardized masfem (mean 0, sd 1)
    - masfem_mturk_z: standardized masfem_mturk (if present)
    - year_centered: year minus mean(year)

    Drops rows missing the key variables required for the primary models.
    """
    df = df.copy()

    # Ensure expected numeric columns exist and coerce types where reasonable
    for col in ['masfem', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'gender_mf']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert source to string/category if present
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Drop rows missing the primary IV or DV or core intensity controls
    required = ['masfem', 'alldeaths', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    present_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=present_required)

    # Create log-transformed outcome variables (use log1p to handle zeros)
    df['log_alldeaths'] = np.log1p(df['alldeaths'].fillna(0))
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'].fillna(0))

    # Standardize masfem (primary IV); create masfem_z
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)

    # If MTurk masculinity/femininity rating is available, standardize for robustness checks
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)

    # Center year to improve interpretability and reduce collinearity
    df['year_centered'] = df['year'] - df['year'].mean()

    # Ensure gender_mf is integer/binary if present
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].astype(pd.Int64Dtype()).astype('float')

    # Keep only columns needed for modeling plus a few originals for reference
    keep_cols = [
        'masfem', 'masfem_z',
        'alldeaths', 'log_alldeaths',
        'ndam15' if 'ndam15' in df.columns else None,
        'log_ndam15' if 'log_ndam15' in df.columns else None,
        'wind', 'min', 'category', 'year', 'year_centered', 'elapsedyrs',
        'gender_mf' if 'gender_mf' in df.columns else None,
        'source' if 'source' in df.columns else None,
        'masfem_mturk_z' if 'masfem_mturk_z' in df.columns else None,
        'name' if 'name' in df.columns else None,
        'ind' if 'ind' in df.columns else None
    ]
    # Remove Nones and duplicates while preserving order
    keep_cols = [c for i, c in enumerate(keep_cols) if c is not None and c not in keep_cols[:i]]

    df = df.loc[:, keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models to test whether more feminine hurricane names are associated
    with downstream adverse outcomes (used here as a proxy for fewer precautions).

    Models run:
    1) OLS on log_alldeaths with HC3 robust standard errors.
    2) OLS on log_ndam15 (log damage) as a secondary outcome with HC3 SEs (if ndam15 present).
    3) Negative Binomial GLM on raw alldeaths as a robustness check for count outcome.

    Returns a dictionary of fitted results objects.
    """
    results = {}
    df = df.copy()

    # Build base covariates list (these must match transformed df columns)
    covariates = ['masfem_z', 'wind', 'min', 'category', 'year_centered', 'elapsedyrs']
    if 'gender_mf' in df.columns:
        covariates.append('gender_mf')

    # Create source dummies if source available (drop first to avoid multicollinearity)
    if 'source' in df.columns:
        src_dummies = pd.get_dummies(df['source'], prefix='source', drop_first=True)
        X = pd.concat([df[covariates], src_dummies], axis=1)
    else:
        X = df[covariates]

    # Add constant
    X_const = sm.add_constant(X, has_constant='add')

    # Primary OLS: log_alldeaths ~ masfem_z + controls
    if 'log_alldeaths' in df.columns:
        y = df['log_alldeaths']
        ols_model = sm.OLS(y, X_const, missing='drop')
        ols_res = ols_model.fit(cov_type='HC3')
        results['ols_deaths'] = ols_res

    # Secondary OLS: log_ndam15 if available
    if 'log_ndam15' in df.columns:
        y2 = df['log_ndam15']
        ols_model2 = sm.OLS(y2, X_const, missing='drop')
        ols_res2 = ols_model2.fit(cov_type='HC3')
        results['ols_damage'] = ols_res2

    # Robustness: Negative Binomial on raw alldeaths (counts)
    # Use GLM NegativeBinomial if alldeaths exists
    if 'alldeaths' in df.columns:
        # ensure non-negative and finite
        y_nb = df['alldeaths'].fillna(0)
        try:
            nb_model = sm.GLM(y_nb, X_const, family=sm.families.NegativeBinomial())
            nb_res = nb_model.fit()
            results['nb_deaths'] = nb_res
        except Exception as e:
            # If NB fails, return the exception string for diagnostics
            results['nb_deaths_error'] = str(e)

    # Return dictionary of models (fitted result objects). Callers can inspect .summary() on each fit.
    return results


