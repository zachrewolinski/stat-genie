from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep rows with the key variables for the analysis
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'year'])

    # Dependent variable transformations
    # Raw death counts (used in count model)
    df['alldeaths'] = df['alldeaths'].astype(float)
    # Log transform for robustness checks (add 1 to handle zeros)
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Standardize continuous predictors (z-scores) for interpretability and numeric stability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / df['masfem'].std()
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / df['wind'].std()
    df['min_z'] = (df['min'] - df['min'].mean()) / df['min'].std()
    df['year_z'] = (df['year'] - df['year'].mean()) / df['year'].std()

    # Categorical encoding for Saffir-Simpson category (create dummies, drop first to serve as baseline)
    # Ensure category is integer-like
    df['category'] = df['category'].astype(int)
    cat_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True)
    df = pd.concat([df, cat_dummies], axis=1)

    # Ensure consistent dummy columns exist even if some categories are absent in the sample
    for c in ['cat_2', 'cat_3', 'cat_4', 'cat_5']:
        if c not in df.columns:
            df[c] = 0

    # Keep only columns that will be referenced in the modeling step plus a few helpful originals
    keep_cols = [
        'ind', 'year', 'name', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'alldeaths', 'log_alldeaths',
        'masfem_z', 'wind_z', 'min_z', 'year_z', 'cat_2', 'cat_3', 'cat_4', 'cat_5'
    ]
    # Some columns might not exist in some input variants, intersect with existing columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs the primary negative binomial model on raw death counts and a robustness OLS on log-deaths.
    Returns a dict with fitted result objects (robust covariance versions).
    """
    # Prepare model covariates (must match transformed column names)
    exog_cols = ['masfem_z', 'wind_z', 'min_z', 'year_z', 'cat_2', 'cat_3', 'cat_4', 'cat_5']

    # Safety: ensure all exog columns exist in df
    for col in exog_cols:
        if col not in df.columns:
            df[col] = 0

    # Build design matrix with a constant
    X = sm.add_constant(df[exog_cols])

    # 1) Primary model: Negative Binomial (appropriate for count data with overdispersion)
    # Use GLM with NegativeBinomial family. We will compute robust (HC3) standard errors.
    nb_model = sm.GLM(df['alldeaths'], X, family=sm.families.NegativeBinomial())
    nb_res = nb_model.fit()
    nb_res_robust = nb_res.get_robustcov_results(cov_type='HC3')

    # 2) Robustness: OLS on log(alldeaths + 1)
    ols_model = sm.OLS(df['log_alldeaths'], X)
    ols_res = ols_model.fit()
    ols_res_robust = ols_res.get_robustcov_results(cov_type='HC3')

    # Return both fitted result objects (robust versions) so the caller can inspect coefficients, std errors, and summaries
    return {
        'negative_binomial_robust': nb_res_robust,
        'ols_log_deaths_robust': ols_res_robust
    }


