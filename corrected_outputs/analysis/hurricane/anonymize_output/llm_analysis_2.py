from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/anonymize_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Simonsohn hurricane dataset into an analysis-ready dataframe.

    Key steps:
    - Rename features to descriptive column names used in modelling.
    - Drop rows missing core variables needed for the primary analysis.
    - Create log-transformed outcome variables (LogDeaths, LogDamage).
    - Standardize the masculinity-femininity index (MasFem_z).
    - Center Year (Year_c) for easier interpretation.
    - Ensure types for categorical / binary columns.

    Returns the transformed dataframe containing the columns referenced in the conceptual variables.
    """
    # Rename columns to descriptive names used in model
    df = df.rename(columns={
        'feature1': 'StormID',
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',
        'feature5': 'MinPressure',
        'feature6': 'FemaleName',
        'feature7': 'Category',
        'feature8': 'Deaths',
        'feature9': 'Damage_Unadj',
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MasFem_MTURK',
        'feature13': 'MaxWind',
        'feature14': 'Damage_2015'
    })

    # Drop rows missing key variables for the primary analysis
    df = df.dropna(subset=['MasFem', 'Deaths', 'MaxWind', 'MinPressure', 'Category', 'Year'])

    # Ensure correct dtypes
    try:
        df['FemaleName'] = df['FemaleName'].astype(int)
    except Exception:
        # If FemaleName is non-numeric, coerce to numeric (map or convert)
        df['FemaleName'] = pd.to_numeric(df['FemaleName'], errors='coerce').fillna(0).astype(int)

    # Category and Year should be integers
    df['Category'] = df['Category'].astype(int)
    df['Year'] = df['Year'].astype(int)

    # Create log-transformed outcomes (add 1 to avoid log(0))
    df['LogDeaths'] = np.log(df['Deaths'].astype(float) + 1.0)

    # Prefer adjusted 2015 damage where available, else fall back to unadjusted damage; finally fill missing with 0
    if 'Damage_2015' in df.columns:
        df['Damage_for_model'] = df['Damage_2015'].fillna(df.get('Damage_Unadj', 0))
    else:
        df['Damage_for_model'] = df.get('Damage_Unadj', 0)
    df['Damage_for_model'] = df['Damage_for_model'].fillna(0).astype(float)
    df['LogDamage'] = np.log(df['Damage_for_model'] + 1.0)

    # Standardize the MasFem index (z-score). Use sample std (ddof=0) for consistent scaling.
    mas_mean = df['MasFem'].mean()
    mas_std = df['MasFem'].std(ddof=0)
    if mas_std == 0 or np.isnan(mas_std):
        df['MasFem_z'] = 0.0
    else:
        df['MasFem_z'] = (df['MasFem'] - mas_mean) / mas_std

    # Center Year for interpretation
    df['Year_c'] = df['Year'] - df['Year'].mean()

    # Keep only columns needed for modeling and for potential robustness checks
    keep_cols = [
        'StormID', 'Year', 'Year_c', 'Name', 'MasFem', 'MasFem_z', 'MasFem_MTURK', 'FemaleName',
        'Category', 'MaxWind', 'MinPressure', 'Deaths', 'LogDeaths', 'Damage_for_model', 'LogDamage', 'Source'
    ]
    # Some columns may not exist in all data versions, so intersect
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].copy()

    # Final sanity checks: drop any rows with infinite or nan in modeling columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['LogDeaths', 'MasFem_z', 'MaxWind', 'MinPressure', 'Category'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit primary and robustness regression models to test whether feminine hurricane names (continuous or binary)
    are associated with higher fatalities (our proxy for fewer precautionary measures) after controlling for
    storm severity and other covariates.

    Primary model: LogDeaths ~ MasFem_z + FemaleName + MaxWind + MinPressure + C(Category) + Year_c
    Robustness model: same specification but predicting LogDamage instead of LogDeaths.

    Returns a dictionary with fitted statsmodels results objects.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist in df
    required = ['LogDeaths', 'MasFem_z', 'FemaleName', 'MaxWind', 'MinPressure', 'Category', 'Year_c']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Primary model: log fatalities
    formula_primary = 'LogDeaths ~ MasFem_z + FemaleName + MaxWind + MinPressure + C(Category) + Year_c'
    primary_res = smf.ols(formula_primary, data=df).fit()

    # Robustness model: log damage
    # Use LogDamage if available; otherwise fall back to Damage_for_model-based log (should be created in transform)
    damage_col = 'LogDamage' if 'LogDamage' in df.columns else ('Damage_for_model' if 'Damage_for_model' in df.columns else None)
    if damage_col is not None and damage_col in df.columns:
        formula_robust = f'{damage_col} ~ MasFem_z + FemaleName + MaxWind + MinPressure + C(Category) + Year_c'
        robust_res = smf.ols(formula_robust, data=df).fit()
    else:
        robust_res = None

    # Return the fitted models; users can call .summary() on each returned result
    return {
        'primary_model': primary_res,
        'robustness_model': robust_res
    }


