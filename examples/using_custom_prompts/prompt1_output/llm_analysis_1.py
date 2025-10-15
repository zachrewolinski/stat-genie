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
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric (coerce invalid -> NaN)
    numeric_cols = ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'wind', 'min', 'category', 'year', 'gender_mf']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep rows with the core variables we need for the main analysis
    required = [c for c in ['alldeaths', 'masfem', 'wind', 'min', 'category'] if c in df.columns]
    df = df.dropna(subset=required)

    # Standardize masfem and masfem_mturk (z-scores). Use population std (ddof=0) for stability with small n.
    if 'masfem' in df.columns:
        df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_z'] = np.nan

    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_mturk_z'] = np.nan

    # Create z-scores for physical intensity components
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1)
    df['min_z'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1)
    df['category_z'] = (df['category'] - df['category'].mean()) / (df['category'].std(ddof=0) if df['category'].std(ddof=0) != 0 else 1)

    # Combine into a single Intensity index: higher wind and higher category increase intensity, lower min pressure increases intensity
    # We combine as wind_z + category_z - min_z and then standardize
    df['Intensity'] = df['wind_z'] + df['category_z'] - df['min_z']
    df['Intensity'] = (df['Intensity'] - df['Intensity'].mean()) / (df['Intensity'].std(ddof=0) if df['Intensity'].std(ddof=0) != 0 else 1)

    # Transform dependent variables: log-transform to reduce skew and accommodate zeros
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Economic damage (robustness): log transform ndam15 if present (fill missing with 0 before log to keep rows)
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['ndam15'] = df['ndam15'].fillna(0)
        df['log_ndam15'] = np.log(df['ndam15'] + 1)
    else:
        df['log_ndam15'] = np.nan

    # Ensure categorical/source is a category dtype for formula handling
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Keep only columns that the model will require to reduce memory and avoid surprises
    keep_cols = ['alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15', 'masfem', 'masfem_z', 'masfem_mturk_z', 'gender_mf', 'wind', 'min', 'category', 'Intensity', 'year', 'source', 'name']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs the primary statistical model and a robustness check.

    Primary model: OLS on log(alldeaths + 1) with the main predictor masfem_z and its interaction with Intensity.
    Controls: gender_mf, categorical source, and year. Robust (HC3) standard errors are used.

    Robustness: same specification but with log_ndam15 (economic damage) as the dependent variable.

    Returns a dict with the fitted results objects so the caller can inspect summaries, coefficients, etc.
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Ensure required columns exist
    required = ['log_alldeaths', 'masfem_z', 'Intensity', 'gender_mf', 'year', 'source']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Main formula: interaction between femininity and intensity
    formula_main = 'log_alldeaths ~ masfem_z * Intensity + gender_mf + C(source) + year'
    res_main = smf.ols(formula_main, data=df).fit(cov_type='HC3')

    # Robustness: economic damage outcome (if available)
    res_robust = None
    if 'log_ndam15' in df.columns and df['log_ndam15'].notnull().any():
        formula_rob = 'log_ndam15 ~ masfem_z * Intensity + gender_mf + C(source) + year'
        res_robust = smf.ols(formula_rob, data=df).fit(cov_type='HC3')

    # Additional optional robustness: use binary gender label instead of continuous masfem (not run by default)
    # formula_bin = 'log_alldeaths ~ gender_mf * Intensity + C(source) + year'
    # res_bin = smf.ols(formula_bin, data=df).fit(cov_type='HC3')

    return {
        'main': res_main,
        'robust_damage': res_robust
    }


