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
    Transform the raw Simonsohn et al. hurricane dataframe into analysis-ready form.

    Produces the following new columns (used in modeling):
      - masfem_z: standardized masfem (higher = more feminine name)
      - masfem_mturk_z: standardized masfem_mturk (alternative IV)
      - FemaleName: binary indicator from gender_mf (0/1)
      - log_deaths: log(alldeaths + 1)
      - log_ndam15: log(ndam15 + 1)
      - wind_z: standardized wind
      - min_z: standardized min pressure
      - Major: 1 if category >= 3, else 0
      - year, category, elapsedyrs, source are kept (source as categorical)

    Rows with missing values in core variables are dropped.
    """
    # make a copy
    df = df.copy()

    # Required columns for analysis
    required = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'elapsedyrs', 'year', 'source']
    missing_req = [c for c in required if c not in df.columns]
    if len(missing_req) > 0:
        raise ValueError(f"Missing required columns in dataframe: {missing_req}")

    # Drop rows with missing values in core variables
    df = df.dropna(subset=['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'elapsedyrs', 'year'])

    # Ensure gender_mf is binary 0/1
    # If already 0/1 keep, otherwise coerce to int
    df['FemaleName'] = df['gender_mf'].astype(int)

    # Log-transform outcomes to reduce skew (add 1 to keep zeros)
    df['log_deaths'] = np.log(df['alldeaths'] + 1)
    df['log_ndam15'] = np.log(df['ndam15'] + 1)

    # Standardize continuous IVs and controls
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
    else:
        df['masfem_mturk_z'] = np.nan

    df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1.0)
    df['min_z'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1.0)

    # Major hurricane indicator
    df['Major'] = (df['category'] >= 3).astype(int)

    # Ensure categorical source (keep original values but cast to category for modeling)
    df['source'] = df['source'].astype('category')

    # Keep only rows with non-missing outcomes after transforms
    df = df.dropna(subset=['log_deaths', 'log_ndam15', 'masfem_z', 'wind_z', 'min_z'])

    # Final column list (helps downstream code know what's available)
    final_cols = ['masfem_z', 'masfem_mturk_z', 'FemaleName', 'alldeaths', 'ndam15', 'log_deaths', 'log_ndam15',
                  'wind_z', 'min_z', 'category', 'Major', 'elapsedyrs', 'year', 'source']

    # If some final columns are missing (e.g., ndam15), ensure they exist (NaN) to avoid KeyErrors later
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit statistical models that test whether more feminine hurricane names are associated with
    changes in proxy measures of public precaution (fatalities and damage).

    Two complementary models are fit:
      1) OLS on log-deaths (log_deaths) with robust standard errors.
      2) Negative binomial GLM on raw death counts (alldeaths) as a count-based check.

    Both models control for key storm intensity and temporal covariates and include source as a categorical control.

    Returns a dict with fitted results objects: {'ols': ols_res, 'nb': nb_res}
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Check required columns exist
    required = ['log_deaths', 'alldeaths', 'masfem_z', 'FemaleName', 'wind_z', 'min_z', 'category', 'elapsedyrs', 'year', 'source']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare formula for OLS on log-deaths
    # Primary IV is masfem_z (continuous). FemaleName included as a second IV (binary).
    formula_ols = 'log_deaths ~ masfem_z + FemaleName + wind_z + min_z + category + Major + elapsedyrs + year + C(source)'

    ols_model = smf.ols(formula=formula_ols, data=df)
    ols_res = ols_model.fit(cov_type='HC3')  # robust standard errors

    # Negative binomial GLM on raw death counts (alternative model appropriate for counts)
    # Use same covariates; GLM formula interface supports categorical source via C(source)
    formula_nb = 'alldeaths ~ masfem_z + FemaleName + wind_z + min_z + category + Major + elapsedyrs + year + C(source)'
    try:
        nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
    except Exception as e:
        nb_res = None
        print('Negative binomial model failed to converge or run:', e)

    # Secondary specifications (not returned but shown): you could replace masfem_z with masfem_mturk_z
    # or use log_ndam15 as the outcome. Example alternative OLS on log damage:
    # formula_damage = 'log_ndam15 ~ masfem_z + FemaleName + wind_z + min_z + category + Major + elapsedyrs + year + C(source)'
    # damage_res = smf.ols(formula=formula_damage, data=df).fit(cov_type='HC3')

    # Return model results. Callers can inspect ols_res.summary() and nb_res.summary() (if not None).
    results = {
        'ols': ols_res,
        'neg_binomial': nb_res
    }
    return results


