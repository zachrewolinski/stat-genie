from typing import Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Attempt to read dataset at import time if available; if not, leave df as None
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')
except Exception:
    df = None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a modeling-ready dataframe.

    Steps:
    - Keep only rows with non-missing values for the variables used in the primary model.
    - Standardize masfem into masfem_z.
    - Ensure alldeaths is integer and non-missing.
    - Ensure category is numeric (int) and source is treated as categorical string.

    Returns the dataframe with columns used in the model: ['alldeaths','masfem_z','wind','category','year','source'] plus original masfem retained.
    """
    # Make a copy to avoid mutating input
    df = df.copy()

    # Required raw columns
    required_cols = ['alldeaths', 'masfem', 'wind', 'category', 'year', 'source']

    # Drop rows missing any of the required columns (raw)
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where appropriate
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    # Coerce category and year to numeric (may produce NaN which will be dropped)
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'category', 'year', 'source'])

    # Now safely cast category and year to native integer dtype (numpy int64)
    # After the dropna above, conversion to int is safe.
    df['category'] = df['category'].astype('int64')
    df['year'] = df['year'].astype('int64')

    # Clip negative alldeaths (if any) to 0 and ensure integer type
    df['alldeaths'] = df['alldeaths'].clip(lower=0).astype(int)

    # Standardize masfem to z-score for interpretability (population std: ddof=0)
    masfem_std = df['masfem'].std(ddof=0)
    if pd.isna(masfem_std) or masfem_std == 0:
        df['masfem_z'] = 0.0
    else:
        df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / masfem_std

    # Ensure source is string/categorical
    df['source'] = df['source'].astype(str)

    # Final columns (keep originals for reproducibility)
    # Ensure the required final columns exist
    final_cols = ['alldeaths', 'masfem', 'masfem_z', 'wind', 'category', 'year', 'source']
    missing_final = [c for c in final_cols if c not in df.columns]
    if missing_final:
        raise RuntimeError(f"Transform failed to produce required columns: {missing_final}")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression predicting hurricane fatalities (alldeaths)
    from standardized femininity of the hurricane name (masfem_z) controlling for
    wind, category, year, and source. Returns the fitted model object.

    We use a GLM with a Negative Binomial family (log link) because the outcome is
    a count with over-dispersion relative to Poisson.
    """
    # Make sure required columns exist
    req = ['alldeaths', 'masfem_z', 'wind', 'category', 'year', 'source']
    missing = [c for c in req if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Use a formula with category and source treated as categorical variables
    formula = 'alldeaths ~ masfem_z + wind + C(category) + year + C(source)'

    # Fit GLM Negative Binomial
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
    results = model_glm.fit()

    # Attach some robustness checks in the results dict for convenience
    # 1) Print summary to console
    print(results.summary())

    # 2) Also fit an OLS on log(ndam15 + 1) as a secondary robustness check (if ndam15 exists)
    robustness = {}
    if 'ndam15' in df.columns:
        try:
            df_copy = df.copy()
            df_copy['log_ndam15'] = np.log(df_copy['ndam15'].astype(float).clip(lower=0) + 1)
            ols_formula = 'log_ndam15 ~ masfem_z + wind + C(category) + year + C(source)'
            ols_mod = smf.ols(ols_formula, data=df_copy).fit(cov_type='HC3')
            robustness['ols_log_damage'] = ols_mod
        except Exception:
            robustness['ols_log_damage'] = None

    # Return main results and any robustness models
    return {'main_nb_glm': results, 'robustness': robustness}