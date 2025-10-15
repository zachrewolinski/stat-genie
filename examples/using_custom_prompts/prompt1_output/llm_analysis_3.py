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
    Prepare the dataframe for modeling. Creates standardized femininity measure and log outcomes.

    Final dataframe will include at minimum these columns used in the models:
      - masfem_z: standardized masfem (z-score)
      - gender_mf: binary female-name indicator (0/1)
      - log_ndam15: log1p of ndam15 (2015-normalized damage)
      - log_alldeaths: log1p of alldeaths (fatalities)
      - wind, category, min, elapsedyrs, year_c
    """
    df = df.copy()

    # Columns required for analysis
    required = ['masfem', 'gender_mf', 'ndam15', 'alldeaths', 'wind', 'category', 'min', 'year', 'elapsedyrs']

    # Coerce to numeric where appropriate, then drop rows missing required data
    for c in required:
        if c in df.columns:
            # leave string columns (not expected) converted to numeric where possible
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Outcome transforms to reduce skew
    df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Primary independent variables
    # Standardize masfem (z-score). Use population std (ddof=0) for stability.
    df['masfem_z'] = (df['masfem'].astype(float) - df['masfem'].astype(float).mean()) / (df['masfem'].astype(float).std(ddof=0))

    # Ensure binary indicator is integer 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Year centered (helps interpretation / numerical stability)
    df['year_c'] = df['year'].astype(float) - df['year'].astype(float).mean()

    # Keep only columns needed for modeling to avoid inadvertent use of other columns
    keep_cols = ['masfem_z', 'gender_mf', 'log_ndam15', 'log_alldeaths', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'elapsedyrs', 'year_c']
    # Some columns may not exist in edge cases; only keep the intersection
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit several models testing whether more feminine hurricane names predict (proxy) lower precaution as measured by lower damages/fatalities:
      1) OLS on log damage (log_ndam15)
      2) OLS on log fatalities (log_alldeaths)
      3) Negative Binomial GLM on raw death counts (alldeaths) as a robustness check for count outcome

    The regressors include the femininity measures and the control set described in the transform function.

    Returns a dict with fitted model objects (statsmodels results).
    """
    # Expected covariates (must match the transform output)
    exog = ['masfem_z', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year_c']
    # Ensure exog are present
    exog = [c for c in exog if c in df.columns]

    # Add constant
    X = sm.add_constant(df[exog], has_constant='add')

    results = {}

    # 1) OLS on log damage
    if 'log_ndam15' in df.columns:
        y_dam = df['log_ndam15']
        ols_damage = sm.OLS(y_dam, X).fit(cov_type='HC3')
        results['ols_damage'] = ols_damage
    else:
        results['ols_damage'] = None

    # 2) OLS on log fatalities
    if 'log_alldeaths' in df.columns:
        y_death = df['log_alldeaths']
        ols_deaths = sm.OLS(y_death, X).fit(cov_type='HC3')
        results['ols_deaths'] = ols_deaths
    else:
        results['ols_deaths'] = None

    # 3) Negative binomial on raw death counts (robustness for count nature of fatalities)
    if 'alldeaths' in df.columns:
        # Use GLM NegativeBinomial; if it fails, catch and return None
        try:
            nb = sm.GLM(df['alldeaths'], X, family=sm.families.NegativeBinomial()).fit()
            results['nb_deaths'] = nb
        except Exception as e:
            # If NB fails to converge, try Poisson as fallback
            try:
                pois = sm.GLM(df['alldeaths'], X, family=sm.families.Poisson()).fit()
                results['nb_deaths'] = pois
            except Exception:
                results['nb_deaths'] = None
    else:
        results['nb_deaths'] = None

    return results


