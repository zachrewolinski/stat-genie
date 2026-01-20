from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'ndam15']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing key variables used in the main (fatalities) model
    df = df.dropna(subset=['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'year'])

    # Dependent variable: Deaths (count)
    df['Deaths'] = df['alldeaths'].astype(int)

    # Independent variables
    # Standardize masfem (higher = more feminine); use population std (ddof=0) for z-score
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)

    # Binary female name indicator (ensure 0/1)
    df['female_name'] = df['gender_mf'].astype(int)

    # Controls: center continuous controls to improve interpretability
    df['wind_c'] = df['wind'] - df['wind'].mean()
    df['category_c'] = df['category'] - df['category'].mean()
    df['min_c'] = df['min'] - df['min'].mean()
    df['year_c'] = df['year'] - df['year'].mean()

    # Keep elapsedyrs as provided (already numeric if present)
    if 'elapsedyrs' not in df.columns:
        df['elapsedyrs'] = pd.NA

    # Create a logged damage variable (useful for secondary/robustness checks)
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log_ndam15'] = np.log(df['ndam15'].fillna(0) + 1)
    else:
        df['log_ndam15'] = np.nan

    # Final dataframe contains the transformed columns used in modeling
    required_cols = [
        'Deaths', 'masfem_z', 'female_name', 'wind_c', 'category_c', 'min_c', 'year_c', 'elapsedyrs', 'log_ndam15', 'ndam15'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # Prepare dataframe (assumes transform has been applied)
    df = df.copy()

    # Select predictors and drop any remaining missing rows for modeling
    model_cols = ['masfem_z', 'female_name', 'wind_c', 'category_c', 'min_c', 'year_c', 'elapsedyrs']
    mod_df = df.dropna(subset=['Deaths'] + model_cols)

    # Design matrix
    X = mod_df[model_cols].astype(float)
    X = sm.add_constant(X)
    y = mod_df['Deaths'].astype(float)

    # 1) Negative binomial regression for counts of fatalities (accounts for overdispersion)
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # In case NegativeBinomial fails, fall back to Poisson with robust covariance
        nb_model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type='HC3')

    # 2) OLS regression predicting logged damages as a robustness check (if damage available)
    ols_model = None
    if 'log_ndam15' in mod_df.columns and mod_df['log_ndam15'].notna().any():
        damage_df = mod_df.dropna(subset=['log_ndam15'])
        if len(damage_df) > 0:
            X_d = damage_df[model_cols].astype(float)
            X_d = sm.add_constant(X_d)
            y_d = damage_df['log_ndam15'].astype(float)
            ols_model = sm.OLS(y_d, X_d).fit()

    # Return model results objects (statsmodels results). Caller can examine summary(), params, pvalues, etc.
    results = {
        'nb_model': nb_model,
        'ols_damage': ols_model
    }
    return results

