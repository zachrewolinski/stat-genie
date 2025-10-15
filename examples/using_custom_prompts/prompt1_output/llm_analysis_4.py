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
    Transform the raw hurricane dataframe to the analysis-ready dataframe.

    Outputs (columns required for the model):
      - alldeaths: dependent variable (kept as integer count)
      - masfem: independent variable (continuous femininity rating)
      - z_wind, z_min, z_elapsedyrs: standardized continuous controls
      - cat_2, cat_3, cat_4, cat_5: category dummies (category 1 is reference)
      - log_alldeaths: log(1 + alldeaths) added for diagnostics (not used as main DV below)
    """
    # Work on a copy
    df = df.copy()

    # Ensure columns used exist
    required_cols = ['alldeaths', 'masfem', 'wind', 'min', 'elapsedyrs', 'category']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values on core variables
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'elapsedyrs', 'category'])

    # Make sure types are numeric where appropriate
    for col in ['alldeaths', 'masfem', 'wind', 'min', 'elapsedyrs', 'category']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'elapsedyrs', 'category'])

    # Dependent variable: keep raw count; add log(1 + alldeaths) for diagnostics
    df['alldeaths'] = df['alldeaths'].astype(float)
    df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Independent variable: masfem (use as provided). Also create a standardized version for interpretation/robustness if desired
    df['masfem'] = df['masfem'].astype(float)
    df['z_masfem'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Continuous controls: standardize wind, min, elapsedyrs (z-scores)
    df['wind'] = df['wind'].astype(float)
    df['min'] = df['min'].astype(float)
    df['elapsedyrs'] = df['elapsedyrs'].astype(float)

    df['z_wind'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1.0)
    df['z_min'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1.0)
    df['z_elapsedyrs'] = (df['elapsedyrs'] - df['elapsedyrs'].mean()) / (df['elapsedyrs'].std(ddof=0) if df['elapsedyrs'].std(ddof=0) != 0 else 1.0)

    # Categorical control: create dummies for categories 2-5 (category 1 as reference)
    # Ensure category is integer
    df['category'] = df['category'].astype(int)
    for k in [2, 3, 4, 5]:
        colname = f'cat_{k}'
        df[colname] = (df['category'] == k).astype(int)

    # If any of the created columns have missing values (shouldn't), fill with 0
    for c in ['z_wind', 'z_min', 'z_elapsedyrs', 'z_masfem', 'cat_2', 'cat_3', 'cat_4', 'cat_5']:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    # Keep only columns necessary for downstream modeling plus some diagnostics
    keep_cols = ['alldeaths', 'log_alldeaths', 'masfem', 'z_masfem', 'z_wind', 'z_min', 'z_elapsedyrs',
                 'cat_2', 'cat_3', 'cat_4', 'cat_5']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial generalized linear model predicting hurricane fatalities (alldeaths)
    from the femininity rating of the hurricane name (masfem), controlling for storm intensity
    and category. Returns the fitted model results object from statsmodels.

    Primary specification:
      alldeaths ~ masfem + z_wind + z_min + z_elapsedyrs + cat_2 + cat_3 + cat_4 + cat_5

    We use a Negative Binomial GLM to account for count data with overdispersion.
    """
    # Ensure the required columns are present
    exog_cols = ['masfem', 'z_wind', 'z_min', 'z_elapsedyrs', 'cat_2', 'cat_3', 'cat_4', 'cat_5']
    missing = [c for c in exog_cols + ['alldeaths'] if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns for the model: {missing}")

    # Build design matrices
    X = df[exog_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['alldeaths'].astype(float)

    # Fit a Negative Binomial GLM (log link by default)
    try:
        model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        results = model_glm.fit()
    except Exception as e:
        # If GLM NegativeBinomial fails (rare), fall back to Poisson with robust SEs
        model_glm = sm.GLM(y, X, family=sm.families.Poisson())
        results = model_glm.fit(cov_type='HC1')

    # Return the fitted results object. The caller can inspect results.summary() or results.params
    return results


