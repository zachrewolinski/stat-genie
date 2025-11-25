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
    Transform the raw hurricane dataset into a dataframe ready for modeling.

    New / transformed columns created (all included in the returned df):
    - alldeaths_count : integer copy of alldeaths (zeros preserved)
    - alldeaths_log   : log1p(alldeaths_count) for OLS robustness checks
    - damage_log      : log1p(ndam15) (log of inflation/normalized damages)
    - masfem_z        : standardized masfem score (mean 0, sd 1)
    - FemaleName      : integer (0/1) from gender_mf
    - year_center     : year centered at its mean
    - source          : string categorical with missing values filled as 'unknown'

    Rows with missing key predictors/controls are dropped (so models have complete cases).
    """
    df = df.copy()

    # Ensure relevant numeric columns are numeric (coerce errors to NaN)
    numeric_cols = [
        'alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min',
        'ndam15', 'year', 'elapsedyrs'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create count DV and log-transformed DV
    if 'alldeaths' in df.columns:
        # fill missing deaths with 0 if appropriate (archival missing might be actual 0s)
        # but conservatively, we convert only non-null values to counts and leave NaN for missing
        df['alldeaths_count'] = df['alldeaths'].fillna(0).astype(int)
        df['alldeaths_log'] = np.log1p(df['alldeaths_count'])
    else:
        df['alldeaths_count'] = np.nan
        df['alldeaths_log'] = np.nan

    # Damage (log) for control
    if 'ndam15' in df.columns:
        df['damage_log'] = np.log1p(df['ndam15'].fillna(0))
    else:
        df['damage_log'] = np.nan

    # Standardize masfem (continuous femininity rating)
    if 'masfem' in df.columns:
        # Use population std (ddof=0) for standardization to be explicit
        mas_mean = df['masfem'].mean()
        mas_std = df['masfem'].std(ddof=0)
        # If mas_std==0 (degenerate), create zeros
        if pd.isna(mas_std) or mas_std == 0:
            df['masfem_z'] = np.nan
        else:
            df['masfem_z'] = (df['masfem'] - mas_mean) / mas_std
    else:
        df['masfem_z'] = np.nan

    # Binary female name indicator
    if 'gender_mf' in df.columns:
        # ensure 0/1 integer
        df['FemaleName'] = pd.to_numeric(df['gender_mf'], errors='coerce').fillna(0).astype(int)
    else:
        df['FemaleName'] = np.nan

    # Year centering
    if 'year' in df.columns:
        df['year_center'] = df['year'] - df['year'].mean()
    else:
        df['year_center'] = np.nan

    # Ensure source is a string and fill missing
    if 'source' in df.columns:
        df['source'] = df['source'].astype(str).fillna('unknown')
    else:
        df['source'] = 'unknown'

    # Drop rows missing key variables used in modeling (complete-case for core variables)
    required_for_model = [
        'masfem_z', 'FemaleName', 'wind', 'category', 'min',
        'year_center', 'elapsedyrs'
    ]
    # Keep rows where these variables are all present (non-null)
    df = df.dropna(subset=required_for_model)

    # Reset index and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit the main statistical models testing whether more feminine hurricane names are associated
    with higher fatalities (operationalized as a proxy for fewer precautionary behaviors).

    Models:
    - Primary: Negative Binomial regression on the count outcome alldeaths_count
               (appropriate for count data with potential over-dispersion).
    - Robustness: OLS on log-transformed deaths (alldeaths_log).

    Returns a dictionary with the fitted model results objects for further inspection.
    """
    import statsmodels.formula.api as smf

    # Formula: include femininity (continuous), binary female name, and controls for severity,
    # time, and reporting source. C(source) treats source as a categorical control.
    formula = (
        'alldeaths_count ~ masfem_z + FemaleName + wind + category + min + '
        'year_center + elapsedyrs + damage_log + C(source)'
    )

    # Fit Negative Binomial (GLM) for counts
    try:
        nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If NegativeBinomial fails (rare), return the exception string for diagnostics
        nb_model = e

    # Robustness OLS on log-transformed deaths
    try:
        ols_formula = formula.replace('alldeaths_count', 'alldeaths_log')
        ols_model = smf.ols(formula=ols_formula, data=df).fit()
    except Exception as e:
        ols_model = e

    # Return both results so downstream code can examine coefficients, p-values, diagnostics
    results = {
        'nb_model': nb_model,
        'ols_log_model': ols_model
    }
    return results


