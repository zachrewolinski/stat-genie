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
    Transform the raw hurricane dataset to a modeling-ready dataframe.
    Produces z-scored masfem (masfem_z), a binary female indicator (gender_female),
    and ensures required columns are present and types are appropriate.

    Input columns expected (from provided schema):
      - alldeaths, masfem, gender_mf, wind, min, category, elapsedyrs, source

    Returns dataframe including at minimum the columns referenced in the conceptual model:
      ['alldeaths', 'masfem_z', 'gender_female', 'wind', 'min', 'category', 'elapsedyrs', 'source']
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'elapsedyrs', 'source']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing dependent variable or primary IV or key intensity controls
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category'])

    # Convert numeric columns to numeric types (coerce if necessary)
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category'])

    # Create z-scored masfem variable to aid interpretation and reduce scale issues
    masfem_mean = df['masfem'].mean()
    masfem_std = df['masfem'].std(ddof=0)
    if masfem_std == 0 or np.isnan(masfem_std):
        # fallback if no variance
        df['masfem_z'] = df['masfem'] - masfem_mean
    else:
        df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std

    # Binary female indicator from provided gender_mf (0 male, 1 female). If not binary, coerce to 0/1
    df['gender_female'] = pd.to_numeric(df['gender_mf'], errors='coerce').fillna(0).astype(int)

    # Normalize source to string and fill missing
    df['source'] = df['source'].astype(str).fillna('unknown')

    # Keep only the columns needed for modeling plus a small set for diagnostics
    keep_cols = ['alldeaths', 'masfem', 'masfem_z', 'gender_female', 'wind', 'min', 'category', 'elapsedyrs', 'source', 'name', 'year']
    for c in keep_cols:
        if c not in df.columns:
            # if optional col is missing, create placeholder
            df[c] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a Negative Binomial regression predicting hurricane fatalities (alldeaths)
    as a function of name femininity (masfem_z) controlling for storm intensity and
    temporal/source covariates.

    Primary specification:
      alldeaths ~ masfem_z + gender_female + wind + min + elapsedyrs + C(category) + C(source)

    Returns the fitted statsmodels results object for the GLM NegativeBinomial model, and
    additionally returns a Poisson with robust SEs and a linear OLS on log(alldeaths+1)
    as robustness checks in a dict.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure transform() has been called and the necessary columns exist
    required = ['alldeaths', 'masfem_z', 'gender_female', 'wind', 'min', 'category', 'elapsedyrs', 'source']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Remove rows with missing predictors or outcome
    mod_df = df.dropna(subset=required).copy()

    # Primary model: Negative Binomial GLM (log link by default)
    formula = 'alldeaths ~ masfem_z + gender_female + wind + min + elapsedyrs + C(category) + C(source)'
    try:
        nb_model = smf.glm(formula=formula, data=mod_df, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If NegativeBinomial fails to converge, fall back to Poisson with robust SEs
        nb_model = None
        print('NegativeBinomial model failed:', e)

    # Robust Poisson as a robustness check
    poisson_model = smf.glm(formula=formula, data=mod_df, family=sm.families.Poisson()).fit(cov_type='HC3')

    # OLS on log(alldeaths + 1) as another robustness check
    mod_df['log_deaths_plus1'] = np.log(mod_df['alldeaths'] + 1)
    ols_model = smf.ols('log_deaths_plus1 ~ masfem_z + gender_female + wind + min + elapsedyrs + C(category) + C(source)', data=mod_df).fit(cov_type='HC3')

    results = {
        'nb_model': nb_model,            # may be None if failed
        'poisson_robust': poisson_model, # Poisson with robust SEs
        'ols_log_outcome': ols_model,    # OLS on log(deaths+1) with robust SEs
        'model_dataframe': mod_df
    }

    return results


