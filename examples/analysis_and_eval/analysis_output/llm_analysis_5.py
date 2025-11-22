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
    Transform the raw hurricane dataset into the dataframe used for modeling.

    Produces the following new/renamed columns used in the model:
    - Femininity: continuous femininity score (masfem, fallback to masfem_mturk)
    - FemaleNameBinary: original gender_mf (kept for exploratory checks)
    - log_deaths: np.log1p(alldeaths)
    - log_damage: np.log1p(ndam15) (kept for robustness checks)
    - min_pressure: renamed from 'min'
    - year_center: year minus mean(year)

    Drops rows missing essential variables for the main analysis.
    """
    df = df.copy()

    # Create femininity score: prefer archival coder score 'masfem', fall back to 'masfem_mturk'
    if 'masfem' in df.columns and 'masfem_mturk' in df.columns:
        df['Femininity'] = df['masfem'].where(~df['masfem'].isna(), df['masfem_mturk'])
    elif 'masfem' in df.columns:
        df['Femininity'] = df['masfem']
    elif 'masfem_mturk' in df.columns:
        df['Femininity'] = df['masfem_mturk']
    else:
        df['Femininity'] = np.nan

    # Binary female name indicator (0 male, 1 female) - keep for robustness
    if 'gender_mf' in df.columns:
        df['FemaleNameBinary'] = df['gender_mf'].astype(float)
    else:
        df['FemaleNameBinary'] = np.nan

    # Dependent variable: log-transformed deaths
    if 'alldeaths' in df.columns:
        df['log_deaths'] = np.log1p(df['alldeaths'].astype(float))
    else:
        df['log_deaths'] = np.nan

    # Auxiliary DV for robustness: log damage (2015-normalized)
    if 'ndam15' in df.columns:
        df['log_damage'] = np.log1p(df['ndam15'].astype(float))
    else:
        df['log_damage'] = np.nan

    # Rename pressure column for clarity
    if 'min' in df.columns:
        df = df.rename(columns={'min': 'min_pressure'})

    # Year centering to aid interpretation and numerical stability
    if 'year' in df.columns:
        df['year_center'] = df['year'].astype(float) - float(df['year'].astype(float).mean())
    else:
        df['year_center'] = np.nan

    # Ensure numeric types for key controls
    for col in ['wind', 'category', 'min_pressure', 'elapsedyrs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the primary independent variable or primary dependent variable
    required = ['Femininity', 'log_deaths', 'wind', 'category', 'min_pressure', 'year_center', 'elapsedyrs', 'source']
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run OLS of log_deaths on Femininity controlling for physical intensity and time/source controls.

    Returns a dict containing the raw OLS fit and a heteroskedasticity-robust version (HC3).
    """
    df = df.copy()

    # Build design matrix: main predictors + controls
    predictors = ['Femininity', 'wind', 'category', 'min_pressure', 'year_center', 'elapsedyrs']
    missing_predictors = [p for p in predictors if p not in df.columns]
    if missing_predictors:
        raise ValueError(f"Missing required predictor columns: {missing_predictors}")

    X_base = df[predictors]

    # Create dummies for source (drop first to avoid multicollinearity). Keep 'source' column in df per the conceptual mapping.
    if 'source' in df.columns:
        source_dummies = pd.get_dummies(df['source'].astype(str), prefix='source', drop_first=True)
        X = pd.concat([X_base, source_dummies], axis=1)
    else:
        X = X_base

    # Dependent variable
    if 'log_deaths' not in df.columns:
        raise ValueError("Dependent variable 'log_deaths' not found in dataframe")
    y = df['log_deaths']

    # Add constant and fit OLS
    X = sm.add_constant(X, has_constant='add')
    ols_model = sm.OLS(y, X).fit()

    # Robust (HC3) standard errors
    robust_model = ols_model.get_robustcov_results(cov_type='HC3')

    # Return both results for inspection/printing
    return {
        'ols': ols_model,
        'robust': robust_model
    }


