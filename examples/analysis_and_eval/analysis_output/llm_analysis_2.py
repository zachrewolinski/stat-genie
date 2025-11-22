from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into analytic dataframe with derived columns used in modeling.

    Outputs (added columns):
    - log_deaths: np.log1p(alldeaths)
    - log_ndam15: np.log1p(ndam15) (kept for possible robustness but not primary DV)
    - masfem_z: standardized masfem score (z)
    - masfem_mturk_z: standardized masfem_mturk score (z)
    - gender_female: binary indicator copied from gender_mf (0/1)
    - year_c: year centered at sample mean
    - wind_z, category_z, min_z: z-scores of storm physical covariates
    - severity_index: composite severity index (higher = more severe)

    Rows with missing values in the main variables used in the principal analyses are dropped.
    """

    df = df.copy()

    # Ensure numeric conversion where appropriate
    # (if input df has these columns as strings, pandas will coerce or raise)
    for col in ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'gender_mf', 'year', 'wind', 'category', 'min']:
        if col in df.columns:
            # coerce to numeric (errors -> NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Primary dependent variable: log(1 + alldeaths)
    if 'alldeaths' in df.columns:
        df['log_deaths'] = np.log1p(df['alldeaths'])
    else:
        df['log_deaths'] = np.nan

    # Also create log damage (robustness)
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        df['log_ndam15'] = np.nan

    # Standardize masfem and masfem_mturk (z-scores). Use population std (ddof=0) to be explicit.
    if 'masfem' in df.columns:
        masfem_mean = df['masfem'].mean()
        masfem_std = df['masfem'].std(ddof=0)
        # guard against zero std
        if pd.isna(masfem_std) or masfem_std == 0:
            df['masfem_z'] = df['masfem'] - masfem_mean
        else:
            df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std
    else:
        df['masfem_z'] = np.nan

    if 'masfem_mturk' in df.columns:
        mm_mean = df['masfem_mturk'].mean()
        mm_std = df['masfem_mturk'].std(ddof=0)
        if pd.isna(mm_std) or mm_std == 0:
            df['masfem_mturk_z'] = df['masfem_mturk'] - mm_mean
        else:
            df['masfem_mturk_z'] = (df['masfem_mturk'] - mm_mean) / mm_std
    else:
        df['masfem_mturk_z'] = np.nan

    # Binary female name indicator (copy and ensure integer 0/1)
    if 'gender_mf' in df.columns:
        # original encoding: 0 male, 1 female (per schema)
        df['gender_female'] = df['gender_mf'].astype('float').round().fillna(0).astype(int)
    else:
        df['gender_female'] = 0

    # Year centered
    if 'year' in df.columns:
        df['year_c'] = df['year'] - df['year'].mean()
    else:
        df['year_c'] = np.nan

    # Standardize physical storm covariates and build severity index
    # For min (pressure), lower values => stronger storm, so we invert its z-score when composing severity
    for c in ['wind', 'category', 'min']:
        if c in df.columns:
            df[c + '_z'] = (df[c] - df[c].mean()) / (df[c].std(ddof=0) if df[c].std(ddof=0) != 0 else 1)
        else:
            df[c + '_z'] = np.nan

    # severity_index: average of wind_z and category_z and inverted min_z (so higher => more severe)
    df['severity_index'] = df[['wind_z', 'category_z', 'min_z']].apply(lambda row: np.nanmean([row['wind_z'], row['category_z'], -row['min_z']]), axis=1)

    # Drop rows missing the main variables for the primary analysis
    required_for_primary = ['log_deaths', 'masfem_z', 'severity_index', 'year_c']
    # Only drop if columns exist; build list of existing required cols
    required_existing = [c for c in required_for_primary if c in df.columns]
    df = df.dropna(subset=required_existing)

    # Return transformed dataframe containing the new columns and original columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary OLS model and robustness checks.

    Primary model:
      log_deaths ~ masfem_z + severity_index + year_c
    Robustness models:
      - Replace masfem_z with binary gender_female
      - Replace masfem_z with masfem_mturk_z

    Uses heteroskedasticity-robust (HC1) standard errors for inference.

    Returns a dictionary of fitted results objects (statsmodels RegressionResults).
    """

    df = df.copy()

    # Ensure necessary columns are present
    required_cols = ['log_deaths', 'masfem_z', 'severity_index', 'year_c']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column for modeling missing: {c}")

    # Build design matrices
    X_main = df[['masfem_z', 'severity_index', 'year_c']]
    X_main = sm.add_constant(X_main)
    y = df['log_deaths']

    # Fit main model with robust (HC1) standard errors
    model_main = sm.OLS(y, X_main).fit(cov_type='HC1')

    # Robustness 1: binary female name
    if 'gender_female' in df.columns:
        X_bin = df[['gender_female', 'severity_index', 'year_c']]
        X_bin = sm.add_constant(X_bin)
        model_gender_bin = sm.OLS(y, X_bin).fit(cov_type='HC1')
    else:
        model_gender_bin = None

    # Robustness 2: MTurk-based femininity z-score
    if 'masfem_mturk_z' in df.columns:
        X_mturk = df[['masfem_mturk_z', 'severity_index', 'year_c']]
        X_mturk = sm.add_constant(X_mturk)
        model_mturk = sm.OLS(y, X_mturk).fit(cov_type='HC1')
    else:
        model_mturk = None

    # Return results in a dictionary for downstream inspection
    results = {
        'main_masfem_z': model_main,
        'gender_binary': model_gender_bin,
        'masfem_mturk_z': model_mturk
    }

    return results


