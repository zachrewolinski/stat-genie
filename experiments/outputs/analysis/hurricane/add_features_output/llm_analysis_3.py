from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the variables used for modeling.

    Produces the following key columns (exact names used in the model):
      - MasFem_c: centered masfem score (higher = more feminine name, mean-centered)
      - LogAllDeaths: log1p transformed total fatalities (alldeaths)
      - Wind: maximum wind speed at landfall
      - MinPressure: minimum pressure at landfall
      - Category: Saffir-Simpson category (kept as integer/categorical)
      - Year: year of occurrence
      - ElapsedYears: years elapsed since the hurricane (kept as numeric)
      - Gender_MF: binary indicator (0 male-coded name, 1 female-coded name)

    Rows with missing values for the core variables are dropped.
    """
    df = df.copy()

    # Normalize / rename source columns to consistent names used downstream
    rename_map = {
        'masfem': 'MasFem',
        'masfem_mturk': 'MasFem_MTurk',
        'gender_mf': 'Gender_MF',
        'alldeaths': 'AllDeaths',
        'ndam15': 'Damage2015',
        'wind': 'Wind',
        'min': 'MinPressure',
        'category': 'Category',
        'year': 'Year',
        'elapsedyrs': 'ElapsedYears',
        'name': 'Name'
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric types where expected
    numeric_cols = ['MasFem', 'Gender_MF', 'AllDeaths', 'Damage2015', 'Wind', 'MinPressure', 'Category', 'Year', 'ElapsedYears']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the key variables needed for the primary analysis
    required_cols = ['MasFem', 'AllDeaths', 'Wind', 'MinPressure', 'Category', 'Year']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Create dependent variable: log(1 + deaths) to reduce skew and keep zero-death storms
    df['LogAllDeaths'] = np.log1p(df['AllDeaths'].fillna(0))

    # Center the masfem score for interpretability (MasFem might be on ~1-11 scale)
    df['MasFem_c'] = df['MasFem'] - df['MasFem'].mean()

    # Ensure binary gender variable is integer 0/1 if available
    if 'Gender_MF' in df.columns:
        # Some datasets may have it as 0/1 already; coerce to int (missing -> 0/1 kept as NaN previously dropped)
        df['Gender_MF'] = df['Gender_MF'].astype(int)
    else:
        # if not present, create NA-filled column for compatibility
        df['Gender_MF'] = pd.Series(np.nan, index=df.index)

    # Cast Category to integer (keep as categorical in model via C(Category) if desired)
    df['Category'] = df['Category'].astype(int)

    # Keep only the columns that will be used in modeling (and keep original Name for possible robustness checks)
    keep_cols = ['Name', 'MasFem', 'MasFem_c', 'MasFem_MTurk' if 'MasFem_MTurk' in df.columns else None, 'Gender_MF', 'AllDeaths', 'LogAllDeaths', 'Damage2015' if 'Damage2015' in df.columns else None, 'Wind', 'MinPressure', 'Category', 'Year', 'ElapsedYears']
    # filter out Nones
    keep_cols = [c for c in keep_cols if c is not None]

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model testing whether more feminine hurricane names (MasFem_c) are associated
    with changes in observed fatalities (LogAllDeaths), controlling for storm intensity and time.

    Primary specification:
      LogAllDeaths ~ MasFem_c * C(Category) + Wind + MinPressure + Year + ElapsedYears + Gender_MF

    The interaction MasFem_c * C(Category) tests whether the name effect differs by storm category
    (i.e., Category is a moderator of the MasFem effect).

    Returns the fitted statsmodels regression results object (with robust HC3 standard errors).
    """
    import statsmodels.formula.api as smf

    # Verify the required columns exist
    required = ['LogAllDeaths', 'MasFem_c', 'Wind', 'MinPressure', 'Category', 'Year', 'ElapsedYears', 'Gender_MF']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: include categorical Category and its interaction with MasFem_c
    formula = 'LogAllDeaths ~ MasFem_c * C(Category) + Wind + MinPressure + Year + ElapsedYears + Gender_MF'

    # Fit OLS and request heteroskedasticity-robust (HC3) standard errors
    model_res = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Return fitted results object (caller can inspect .summary(), .params, etc.)
    return model_res


