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

    # Rename columns to meaningful names used in modeling
    rename_map = {
        'feature13': 'Deaths',               # total number of deaths
        'feature9': 'MasFeminine',           # masculinity-femininity index (coders)
        'feature11': 'MasFeminineMTurk',     # MTurk masfem index
        'feature12': 'FemaleName',           # binary female name indicator (0 male, 1 female)
        'feature7': 'MaxWind',               # max wind speed
        'feature4': 'Category',              # Saffir-Simpson category
        'feature14': 'MinPressure',          # minimum pressure
        'feature5': 'Year',                  # year of hurricane
        'feature8': 'PropDamage2015',        # property damage adjusted to 2015
        'feature2': 'StormID',
        'feature6': 'Name',
        'feature10': 'Source',
        'feature3': 'YearsSince'
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric types for the key variables
    numeric_cols = ['Deaths', 'MasFeminine', 'MasFeminineMTurk', 'FemaleName', 'MaxWind', 'Category', 'MinPressure', 'Year', 'PropDamage2015']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the primary outcome or primary IVs or essential severity controls
    required_for_model = ['Deaths', 'MasFeminine', 'MaxWind', 'Category', 'MinPressure', 'Year']
    df = df.dropna(subset=required_for_model)

    # Coerce FemaleName to integer 0/1 if present (some datasets already have 0/1)
    if 'FemaleName' in df.columns:
        df['FemaleName'] = df['FemaleName'].apply(lambda x: 1 if x == 1 or str(x).strip() == '1' else (0 if x == 0 or str(x).strip() == '0' else np.nan)).astype('float')

    # Compute log-transformed dependent variable: log(Deaths + 1)
    df['LogDeaths'] = np.log(df['Deaths'].fillna(0) + 1)

    # Compute log property damage for robustness checks (handle zeros safely)
    if 'PropDamage2015' in df.columns:
        df['PropDamage2015'] = pd.to_numeric(df['PropDamage2015'], errors='coerce')
        df['PropDamage2015'] = df['PropDamage2015'].replace({0: np.nan})
        df['LogPropDamage'] = np.log(df['PropDamage2015'].fillna(0) + 1)

    # Standardize the continuous masfem indices (z-score) so coefficients are interpretable
    df['MasFeminine_z'] = (df['MasFeminine'] - df['MasFeminine'].mean()) / (df['MasFeminine'].std(ddof=0) if df['MasFeminine'].std(ddof=0) != 0 else 1)
    if 'MasFeminineMTurk' in df.columns:
        df['MasFeminineMTurk_z'] = (df['MasFeminineMTurk'] - df['MasFeminineMTurk'].mean()) / (df['MasFeminineMTurk'].std(ddof=0) if df['MasFeminineMTurk'].std(ddof=0) != 0 else 1)

    # Keep only columns necessary for modeling and diagnostics
    cols_keep = [c for c in ['StormID', 'Name', 'Source', 'Deaths', 'LogDeaths', 'MasFeminine', 'MasFeminine_z', 'MasFeminineMTurk', 'MasFeminineMTurk_z', 'FemaleName', 'MaxWind', 'Category', 'MinPressure', 'Year', 'PropDamage2015', 'LogPropDamage', 'YearsSince'] if c in df.columns]
    df = df[cols_keep]

    # Final drop of any rows with NA in DV or the main IV after transformations
    df = df.dropna(subset=['LogDeaths', 'MasFeminine_z', 'MaxWind', 'Category', 'MinPressure', 'Year'])

    # Ensure Category is integer-like for categorical treatment in formulas
    df['Category'] = df['Category'].astype('Int64')

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """Runs primary and robustness OLS regressions to estimate the association
    between name femininity and hurricane fatalities (as a proxy for precautionary behavior).

    Returns a dictionary of fitted statsmodels results objects.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Main model: continuous standardized masfem index controlling for physical severity and year.
    # Treat Category as categorical with C(Category).
    formula_main = 'LogDeaths ~ MasFeminine_z + MaxWind + C(Category) + MinPressure + Year'
    model_main = smf.ols(formula_main, data=df).fit(cov_type='HC3')
    results['main_masfem_continuous'] = model_main

    # Alternative: use binary female-name indicator instead of continuous masfem
    if 'FemaleName' in df.columns and df['FemaleName'].notnull().all():
        formula_bin = 'LogDeaths ~ FemaleName + MaxWind + C(Category) + MinPressure + Year'
        model_bin = smf.ols(formula_bin, data=df).fit(cov_type='HC3')
        results['alt_female_binary'] = model_bin

    # Robustness: use MTurk masfem index if available
    if 'MasFeminineMTurk_z' in df.columns:
        formula_mturk = 'LogDeaths ~ MasFeminineMTurk_z + MaxWind + C(Category) + MinPressure + Year'
        model_mturk = smf.ols(formula_mturk, data=df).fit(cov_type='HC3')
        results['robust_mturk_masfem'] = model_mturk

    # Return the fitted model objects so the caller can inspect summaries, coefficients, etc.
    return results


