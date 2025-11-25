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
    Transform raw hurricane dataframe into analysis-ready dataframe.

    Produces the following final columns used in modeling (and documented in cvars):
      - masfem_z : z-scored continuous femininity rating (IV)
      - gender_mf : binary female name indicator (0/1)
      - alldeaths : count outcome (DV)
      - min_pressure : renamed 'min' (numeric)
      - max_wind : renamed 'wind' (numeric)
      - category : Saffir-Simpson category (numeric)
      - damage_2015 : renamed 'ndam15' (numeric)
      - log_damage_2015 : log1p(damage_2015)
      - year_center : year - mean(year)
      - elapsedyrs : as provided
      - source : categorical source indicator (kept as original string/category)

    Rows with missing critical values are dropped.
    """
    df = df.copy()

    # Keep only rows with necessary fields
    required_cols = ['masfem', 'alldeaths', 'min', 'wind', 'category', 'ndam15', 'year', 'elapsedyrs', 'source', 'gender_mf']
    # If some of these columns do not exist in the input, this will raise a KeyError so the user is alerted.
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values in critical variables
    df = df.dropna(subset=['masfem', 'alldeaths', 'min', 'wind', 'category', 'ndam15', 'year', 'elapsedyrs'])

    # Rename columns for clarity
    df = df.rename(columns={
        'min': 'min_pressure',
        'wind': 'max_wind',
        'ndam15': 'damage_2015'
    })

    # Ensure numeric types
    numeric_cols = ['masfem', 'alldeaths', 'min_pressure', 'max_wind', 'category', 'damage_2015', 'year', 'elapsedyrs', 'gender_mf']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop any rows that turned NA after coercion
    df = df.dropna(subset=numeric_cols)

    # Create standardized masfem (z-score) for interpretation and numerical stability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Transform damage to reduce skew (use log1p to handle zeros)
    df['log_damage_2015'] = np.log1p(df['damage_2015'].astype(float))

    # Center year to aid interpretation
    df['year_center'] = df['year'] - df['year'].mean()

    # Ensure alldeaths is an integer count (non-negative). If negative values exist they are suspicious; coerce to zero minimum.
    df['alldeaths'] = df['alldeaths'].astype(float).clip(lower=0).astype(int)

    # Ensure source is string / categorical and fill missing as 'unknown'
    df['source'] = df['source'].fillna('unknown').astype(str)

    # Keep only columns we will use in modeling (plus a few originals for robustness)
    keep_cols = [
        'masfem', 'masfem_z', 'gender_mf', 'alldeaths', 'min_pressure', 'max_wind',
        'category', 'damage_2015', 'log_damage_2015', 'year', 'year_center', 'elapsedyrs', 'source'
    ]
    df = df[keep_cols]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial generalized linear model predicting hurricane fatalities (alldeaths)
    from the standardized femininity rating of the name (masfem_z) controlling for storm
    intensity and other covariates.

    Returns the fitted GLMResults object.
    """
    df = df.copy()

    # Construct design matrix
    # Use masfem_z as the main IV. Include gender_mf as auxiliary control.
    X_parts = [
        df[['masfem_z', 'gender_mf', 'min_pressure', 'max_wind', 'category', 'log_damage_2015', 'year_center', 'elapsedyrs']]
    ]

    # Expand source into dummy variables (drop first to avoid multicollinearity)
    source_dummies = pd.get_dummies(df['source'].astype(str), prefix='source', drop_first=True)
    if not source_dummies.empty:
        X_parts.append(source_dummies)

    X = pd.concat(X_parts, axis=1)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Outcome
    y = df['alldeaths']

    # Fit Negative Binomial GLM with log link (appropriate for overdispersed count data)
    # If the NegativeBinomial family is unavailable or fails, the user can try Poisson as fallback.
    try:
        model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        results = model_nb.fit()
    except Exception as e:
        # Fallback to Poisson with robust covariance
        model_pois = sm.GLM(y, X, family=sm.families.Poisson())
        results = model_pois.fit(cov_type='HC3')

    # Return the fitted results object (caller can inspect results.summary())
    return results


