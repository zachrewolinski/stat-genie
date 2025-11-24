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
    # Work on a copy
    df = df.copy()

    # Ensure relevant numeric columns are numeric
    numeric_cols = ['alldeaths', 'masfem', 'gender_mf', 'ndam15', 'wind', 'category', 'min', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the primary variables needed for analysis
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind'])

    # Dependent variable: log(1 + alldeaths) to reduce skew and handle zeros
    df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Secondary dependent/control transform: log damage (2015 dollars) if available
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        df['log_ndam15'] = np.nan

    # Binary female name indicator (ensure 0/1)
    if 'gender_mf' in df.columns:
        df['GenderFemale'] = df['gender_mf'].astype(float).round().astype('Int64')
    else:
        df['GenderFemale'] = pd.Series([pd.NA] * len(df), index=df.index)

    # Standardize the main IV (masfem) and the moderator (wind) using population std (ddof=0)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / df['masfem'].std(ddof=0)
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / df['wind'].std(ddof=0)

    # Interaction between standardized masfem and standardized wind (for moderation test)
    df['masfem_wind'] = df['masfem_z'] * df['wind_z']

    # Keep only columns needed for the model and return full dataframe so users can inspect other variables if desired
    # (the model function will drop any rows with remaining missing model variables)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    # Copy to avoid mutating original
    df_m = df.copy()

    # Ensure model columns exist and drop rows with missing model data
    model_cols = ['log_alldeaths', 'masfem_z', 'GenderFemale', 'wind_z', 'masfem_wind', 'min', 'category', 'year', 'log_ndam15', 'alldeaths']
    existing_cols = [c for c in model_cols if c in df_m.columns]
    df_m = df_m.dropna(subset=['log_alldeaths', 'masfem_z', 'wind_z'])

    # Prepare design matrix X and outcome y for the primary OLS on log deaths
    X_cols = ['masfem_z', 'GenderFemale', 'wind_z', 'masfem_wind', 'min', 'category', 'year', 'log_ndam15']
    X_cols = [c for c in X_cols if c in df_m.columns]
    X = df_m[X_cols].astype(float)
    X = sm.add_constant(X)
    y = df_m['log_alldeaths'].astype(float)

    # Primary model: OLS on log(1 + deaths) with heteroskedasticity-consistent SEs (HC3)
    ols_res = sm.OLS(y, X).fit(cov_type='HC3')

    # Robustness: Negative binomial on raw counts (alldeaths) to respect count nature (if 'alldeaths' present)
    nb_res = None
    if 'alldeaths' in df_m.columns:
        try:
            # Use same covariates (without log transformed y) for count model
            y_count = df_m['alldeaths'].astype(float)
            # For GLM, do not include log_ndam15 if it is NaN for all or missing; already filtered above
            nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial())
            nb_res = nb_model.fit()
        except Exception:
            nb_res = None

    # Return fitted results objects so the caller can inspect summaries, params, etc.
    return {
        'ols_result': ols_res,
        'nb_result': nb_res
    }


