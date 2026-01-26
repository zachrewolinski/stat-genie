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
    Prepare the hurricane dataset for modeling.

    Produces the following transformed columns used by the statistical models:
      - masfem_z: standardized masfem score (mean 0, sd 1)
      - log_alldeaths: log(alldeaths + 1)
      - log_ndam15: log(ndam15 + 1) if ndam15 exists

    Also coerces key control variables to numeric types so downstream modeling is robust.
    """
    df = df.copy()

    # Ensure numeric conversion for relevant columns (safe coercion)
    numeric_cols = [
        'masfem', 'masfem_mturk', 'gender_mf', 'wind', 'category', 'min',
        'alldeaths', 'ndam', 'ndam15', 'elapsedyrs', 'year'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the primary independent variable or primary dependent variable
    if 'masfem' in df.columns and 'alldeaths' in df.columns:
        df = df.dropna(subset=['masfem', 'alldeaths'])

    # Create log-transformed dependent variables
    if 'alldeaths' in df.columns:
        # log(0) undefined -> use log1p for zeros
        df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))
    else:
        df['log_alldeaths'] = np.nan

    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))
    else:
        # keep column present for consistency
        df['log_ndam15'] = np.nan

    # Standardize the masfem score (z-score) so coefficients are interpretable
    if 'masfem' in df.columns:
        masfem_mean = df['masfem'].mean()
        masfem_std = df['masfem'].std(ddof=0)
        if pd.isna(masfem_std) or masfem_std == 0:
            # fallback: keep raw if no variation
            df['masfem_z'] = df['masfem']
        else:
            df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std
    else:
        df['masfem_z'] = np.nan

    # Ensure gender_mf is integer 0/1 when present (preserve NaN if missing)
    if 'gender_mf' in df.columns:
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')
        # Round values that may be floats representing 0/1
        df.loc[df['gender_mf'].notnull(), 'gender_mf'] = df.loc[df['gender_mf'].notnull(), 'gender_mf'].round().astype(int)

    # Ensure categorical/int controls are numeric types
    for col in ['wind', 'category', 'min', 'elapsedyrs', 'year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Final: keep original columns plus engineered ones. Return the transformed dataframe.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS models predicting log-transformed fatalities (primary) and log damages (secondary).

    Primary specification:
      log_alldeaths ~ masfem_z + gender_mf + masfem_z:gender_mf + wind + category + min + elapsedyrs

    Models are estimated with HC3 robust standard errors.

    Returns: dict with keys 'death_model' and 'damage_model' (the latter may be None if ndam15 is not available).
    """
    results = {}
    df = df.copy()

    # Primary model: fatalities
    required_cols = ['log_alldeaths', 'masfem_z', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs']
    present = all(col in df.columns for col in required_cols)
    if not present:
        raise ValueError(f"One or more required columns for the primary model are missing: {required_cols}")

    df_death = df[required_cols].dropna()
    # Design matrix with interaction term between standardized masfem and gender indicator
    X_death = pd.DataFrame({
        'masfem_z': df_death['masfem_z'],
        'gender_mf': df_death['gender_mf'],
        'masfem_x_gender': df_death['masfem_z'] * df_death['gender_mf'],
        'wind': df_death['wind'],
        'category': df_death['category'],
        'min': df_death['min'],
        'elapsedyrs': df_death['elapsedyrs']
    })
    X_death = sm.add_constant(X_death)
    y_death = df_death['log_alldeaths']

    death_model = sm.OLS(y_death, X_death).fit(cov_type='HC3')
    results['death_model'] = death_model

    # Secondary model: damages (ndam15) if available
    if 'log_ndam15' in df.columns:
        required_damage_cols = ['log_ndam15', 'masfem_z', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs']
        if all(col in df.columns for col in required_damage_cols):
            df_damage = df[required_damage_cols].dropna()
            if len(df_damage) > 0:
                X_damage = pd.DataFrame({
                    'masfem_z': df_damage['masfem_z'],
                    'gender_mf': df_damage['gender_mf'],
                    'masfem_x_gender': df_damage['masfem_z'] * df_damage['gender_mf'],
                    'wind': df_damage['wind'],
                    'category': df_damage['category'],
                    'min': df_damage['min'],
                    'elapsedyrs': df_damage['elapsedyrs']
                })
                X_damage = sm.add_constant(X_damage)
                y_damage = df_damage['log_ndam15']
                damage_model = sm.OLS(y_damage, X_damage).fit(cov_type='HC3')
                results['damage_model'] = damage_model
            else:
                results['damage_model'] = None
        else:
            results['damage_model'] = None
    else:
        results['damage_model'] = None

    return results


