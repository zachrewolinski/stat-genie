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
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure numeric columns are numeric
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')
    df['masfem_mturk'] = pd.to_numeric(df.get('masfem_mturk'), errors='coerce')

    # Drop rows missing the core IV or DV
    df = df.dropna(subset=['masfem', 'alldeaths'])

    # Create a standardized masfem variable for ease of interpretation
    df['masfem_std'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Binary female name indicator (gender_mf is 1 for female, 0 for male). If missing, try to infer from masfem (> median -> female) as fallback
    df['FemaleName'] = df['gender_mf']
    # If gender_mf missing, infer using masfem median split
    missing_gender = df['FemaleName'].isna()
    if missing_gender.any():
        median_m = df['masfem'].median()
        df.loc[missing_gender, 'FemaleName'] = (df.loc[missing_gender, 'masfem'] > median_m).astype(int)
    df['FemaleName'] = df['FemaleName'].astype(int)

    # Dependent variable transformations: log(1 + alldeaths) to reduce skew and deal with zeros
    df['log_alldeaths'] = np.log1p(df['alldeaths'].fillna(0))

    # Also prepare a log-transformed damage variable for robustness checks
    df['log_ndam15'] = np.log1p(df['ndam15'].fillna(0))

    # Create source dummies (drop_first=True to avoid multicollinearity). Keep column names stable.
    df['source'] = df['source'].astype(str).fillna('unknown')
    source_dummies = pd.get_dummies(df['source'], prefix='source', drop_first=True)
    # Concatenate dummies into df
    df = pd.concat([df, source_dummies], axis=1)

    # Ensure other numeric controls have no missing values where possible (drop rows with missing essential controls)
    # For the main specification we require wind, category, and min (storm intensity). If missing, drop.
    df = df.dropna(subset=['wind', 'category', 'min', 'year', 'elapsedyrs'])

    # Final columns expected by the modeling function (kept in dataframe):
    # 'masfem', 'masfem_std', 'FemaleName', 'gender_mf', 'alldeaths', 'log_alldeaths',
    # 'ndam15', 'log_ndam15', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'masfem_mturk', plus source dummies

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # Make a local copy
    df = df.copy()

    # Identify source dummy columns created in transform (those that start with 'source_')
    source_dummy_cols = [c for c in df.columns if c.startswith('source_')]

    # Primary OLS model: log fatalities on masfem (standardized) controlling for storm intensity and time
    y = df['log_alldeaths']

    control_cols = ['wind', 'min', 'category', 'year', 'elapsedyrs']
    # Include masfem_mturk as an additional control if available (robustness)
    if 'masfem_mturk' in df.columns:
        control_cols = control_cols + ['masfem_mturk']

    X_cols = ['masfem_std'] + control_cols + source_dummy_cols

    # Filter X to available columns
    X = df[X_cols].copy()
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust (HC3) standard errors
    ols_model = sm.OLS(y, X).fit(cov_type='HC3')

    # Robustness 1: Negative Binomial regression on raw counts (alldeaths) to account for count nature and overdispersion
    # Use the same regressors (without the log transform on y)
    try:
        nb_model = sm.GLM(df['alldeaths'], X, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        nb_model = None

    # Robustness 2: Alternative DV - log of damage
    if 'log_ndam15' in df.columns:
        y_damage = df['log_ndam15']
        ols_damage = sm.OLS(y_damage, X).fit(cov_type='HC3')
    else:
        ols_damage = None

    # Return a dictionary of fitted results for inspection
    results = {
        'ols_log_deaths': ols_model,
        'neg_binom_deaths': nb_model,
        'ols_log_damage': ols_damage,
        'X_columns': X.columns.tolist()
    }

    return results


