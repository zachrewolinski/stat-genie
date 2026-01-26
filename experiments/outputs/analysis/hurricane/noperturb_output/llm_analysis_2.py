from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a modeling-ready dataframe.

    Produces these derived columns (all included in the returned df):
    - log_ndam15: np.log1p(ndam15)  (primary DV)
    - log_ndam: np.log1p(ndam)      (alternative DV, kept for robustness)
    - log_alldeaths: np.log1p(alldeaths)  (alternative DV: fatalities)
    - FemaleName: binary indicator equal to gender_mf (0/1)
    - masfem_z: standardized masfem (mean=0, sd=1)

    Also ensures required control columns exist and drops rows with missing values in variables used in the primary model.
    """

    # Copy to avoid modifying input in-place
    df = df.copy()

    # Standardize column names expected (if any stray whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric types where expected
    numeric_cols = ['masfem', 'gender_mf', 'ndam15', 'ndam', 'alldeaths', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Primary dependent variable: inflation-adjusted damage (2015) -> log transform
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        # if ndam15 absent, try ndam
        if 'ndam' in df.columns:
            df['log_ndam15'] = np.log1p(df['ndam'])
        else:
            df['log_ndam15'] = np.nan

    # Alternative DVs for robustness
    if 'ndam' in df.columns:
        df['log_ndam'] = np.log1p(df['ndam'])
    else:
        df['log_ndam'] = np.nan
    if 'alldeaths' in df.columns:
        df['log_alldeaths'] = np.log1p(df['alldeaths'].fillna(0))
    else:
        df['log_alldeaths'] = np.nan

    # Independent variables
    # masfem: keep as-is and create standardized version for interpretability
    if 'masfem' in df.columns:
        df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) > 0 else 1.0)
    else:
        df['masfem_z'] = np.nan

    # Binary female name indicator (0/1)
    if 'gender_mf' in df.columns:
        # ensure values are 0/1
        df['FemaleName'] = df['gender_mf'].map({0: 0, 1: 1}).astype('float')
    else:
        df['FemaleName'] = np.nan

    # Source as categorical (keep original values but fill NAs)
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category').fillna('unknown')
    else:
        df['source'] = 'unknown'

    # Drop rows with missing data for primary analysis: require masfem (or gender), and ndam15 and key controls
    required_for_primary = ['log_ndam15', 'masfem_z', 'FemaleName', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'source']
    present_required = [c for c in required_for_primary if c in df.columns]
    df = df.dropna(subset=present_required)

    # Optional: Cap extremely large ndam15 values to reduce undue leverage (winsorize top 1%)
    if 'log_ndam15' in df.columns:
        upper = df['log_ndam15'].quantile(0.99)
        df['log_ndam15_winsor'] = np.where(df['log_ndam15'] > upper, upper, df['log_ndam15'])
    else:
        df['log_ndam15_winsor'] = np.nan

    # Return transformed dataframe. Columns required by the model are: masfem, masfem_z, FemaleName,
    # log_ndam15 (and winsorized), wind, category, min, year, elapsedyrs, source, plus alternatives retained.
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary OLS models that test whether more feminine hurricane names are associated with greater damages
    (interpreted as less precautionary behavior). Returns a dictionary with fitted models and summaries.

    Primary model: log_ndam15 ~ masfem_z + FemaleName + wind + category + min + year + elapsedyrs + C(source)
    Robust (HC3) standard errors are used.

    Also fit robustness models using log_alldeaths and using the winsorized log_ndam15.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Ensure required columns exist
    required = ['log_ndam15', 'masfem_z', 'FemaleName', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'source']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Primary OLS model (damage, continuous masfem and binary female indicator included together)
    formula_primary = 'log_ndam15 ~ masfem_z + FemaleName + wind + category + min + year + elapsedyrs + C(source)'
    model_primary = smf.ols(formula_primary, data=df).fit(cov_type='HC3')
    results['model_primary'] = model_primary

    # Robustness 1: winsorized damage
    if 'log_ndam15_winsor' in df.columns:
        formula_win = 'log_ndam15_winsor ~ masfem_z + FemaleName + wind + category + min + year + elapsedyrs + C(source)'
        model_win = smf.ols(formula_win, data=df).fit(cov_type='HC3')
        results['model_winsor'] = model_win

    # Robustness 2: fatalities as outcome (if present)
    if 'log_alldeaths' in df.columns and df['log_alldeaths'].notna().any():
        formula_deaths = 'log_alldeaths ~ masfem_z + FemaleName + wind + category + min + year + elapsedyrs + C(source)'
        model_deaths = smf.ols(formula_deaths, data=df).fit(cov_type='HC3')
        results['model_deaths'] = model_deaths

    # Return dictionary of fitted models. Each value is a statsmodels RegressionResultsWrapper
    return results

