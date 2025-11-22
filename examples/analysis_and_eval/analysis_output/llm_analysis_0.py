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
    Transform the raw hurricane dataset into a dataframe ready for modeling.

    Produces these final columns (used by the model):
      - LogDeaths: log(alldeaths + 1)
      - LogDamage: log(ndam15 + 1) [kept for robustness]
      - masfem_z: standardized masfem score (main IV)
      - masfem_mturk_z: standardized masfem_mturk score (alternative IV)
      - gender_female: integer 0/1 indicator from gender_mf
      - wind_z: standardized wind
      - min_z: standardized min (pressure)
      - year_center: year - mean(year)
      - cat_2, cat_3, cat_4, cat_5: explicit category dummies (1/0); category 1 is the implicit reference

    Rows missing the primary outcome or primary IV or core intensity controls are dropped.
    """
    # Make a copy to avoid mutating input
    df = df.copy()

    # Ensure numeric columns are numeric
    for col in ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'gender_mf', 'wind', 'min', 'year', 'category']:
        if col in df.columns:
            # coerce errors -> NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Primary outcome: total deaths (log-transformed). Keep ndam15 log for robustness.
    if 'alldeaths' in df.columns:
        df['LogDeaths'] = np.log(df['alldeaths'].fillna(0) + 1)
    else:
        df['LogDeaths'] = np.nan
    if 'ndam15' in df.columns:
        df['LogDamage'] = np.log(df['ndam15'].fillna(0) + 1)
    else:
        df['LogDamage'] = np.nan

    # Independent variables: femininity scores and binary gender indicator
    if 'masfem' in df.columns:
        df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_z'] = np.nan
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_mturk_z'] = np.nan
    if 'gender_mf' in df.columns:
        # Ensure 0/1 integer
        df['gender_female'] = df['gender_mf'].fillna(0).astype(int)
    else:
        df['gender_female'] = np.nan

    # Controls: standardize continuous intensity measures and center year
    if 'wind' in df.columns:
        df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1)
    else:
        df['wind_z'] = np.nan
    if 'min' in df.columns:
        # Lower pressure = stronger storm; keep as control
        df['min_z'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1)
    else:
        df['min_z'] = np.nan
    if 'year' in df.columns:
        df['year_center'] = df['year'] - df['year'].mean()
    else:
        df['year_center'] = np.nan

    # Create explicit category indicator variables (category 1 is implicit reference)
    # We create cat_2..cat_5 as 0/1 columns so model matrix is predictable.
    for c in [2, 3, 4, 5]:
        df[f'cat_{c}'] = 0
    if 'category' in df.columns:
        # If category has non-integer values, coerce to int where possible
        df['category'] = pd.to_numeric(df['category'], errors='coerce')
        for c in [2, 3, 4, 5]:
            df.loc[df['category'] == c, f'cat_{c}'] = 1

    # Drop rows that are missing the primary outcome or the primary IV or core intensity controls
    required_for_main = ['LogDeaths', 'masfem_z', 'wind_z', 'min_z', 'year_center']
    present_required = [c for c in required_for_main if c in df.columns]
    df = df.dropna(subset=present_required)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS models to test whether storms with more feminine names (masfem) are associated
    with higher human cost (LogDeaths) controlling for storm intensity and year.

    Returns a dictionary with:
      - 'main_model': OLS fit using continuous masfem_z
      - 'gender_model': OLS fit using binary gender_female instead of masfem_z (robustness)
      - 'damage_model': OLS fit using LogDamage as outcome (robustness)

    Each model controls for wind_z, min_z, year_center and category dummies cat_2..cat_5.
    """
    # Ensure statsmodels is available
    import statsmodels.api as sm

    # Columns used in main model
    control_cols = [c for c in ['wind_z', 'min_z', 'year_center', 'cat_2', 'cat_3', 'cat_4', 'cat_5'] if c in df.columns]

    results = {}

    # Main model: LogDeaths ~ masfem_z + controls
    if ('LogDeaths' in df.columns) and ('masfem_z' in df.columns):
        X_main_cols = ['masfem_z'] + control_cols
        X_main = df[X_main_cols].astype(float)
        X_main = sm.add_constant(X_main)
        y_main = df['LogDeaths'].astype(float)
        model_main = sm.OLS(y_main, X_main, missing='drop').fit()
        results['main_model'] = model_main
    else:
        results['main_model'] = None

    # Robustness 1: use binary gender indicator instead of continuous masfem
    if ('LogDeaths' in df.columns) and ('gender_female' in df.columns):
        X_g_cols = ['gender_female'] + control_cols
        X_g = df[X_g_cols].astype(float)
        X_g = sm.add_constant(X_g)
        y_g = df['LogDeaths'].astype(float)
        model_gender = sm.OLS(y_g, X_g, missing='drop').fit()
        results['gender_model'] = model_gender
    else:
        results['gender_model'] = None

    # Robustness 2: outcome = LogDamage (property damage) to see if pattern holds for economic impact
    if ('LogDamage' in df.columns) and ('masfem_z' in df.columns):
        X_d_cols = ['masfem_z'] + control_cols
        X_d = df[X_d_cols].astype(float)
        X_d = sm.add_constant(X_d)
        y_d = df['LogDamage'].astype(float)
        model_damage = sm.OLS(y_d, X_d, missing='drop').fit()
        results['damage_model'] = model_damage
    else:
        results['damage_model'] = None

    # Return the fitted model objects so the caller can inspect .summary(), coefficients, CIs, etc.
    return results


