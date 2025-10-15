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
    # Make a copy to avoid modifying caller's dataframe
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['masfem', 'masfem_mturk', 'min', 'category', 'alldeaths', 'ndam15', 'wind', 'elapsedyrs', 'gender_mf']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing key variables needed for the main model
    required = ['masfem', 'alldeaths', 'wind', 'min', 'category', 'elapsedyrs']
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Dependent variable: log(1 + total deaths) to reduce skew and handle zeros
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Independent variable: center masfem for interpretability
    df['masfem_c'] = df['masfem'].astype(float) - df['masfem'].astype(float).mean()

    # Include alternative IV (binary name-gender) kept as-is for robustness checks
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].astype(int)

    # Ensure categorical/ordinal variable 'category' is numeric (already numeric in schema)
    df['category'] = pd.to_numeric(df['category'], errors='coerce')

    # Optionally create a simple severity index (z-scored) combining wind and inverse pressure
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1)
    df['min_inv_z'] = ((-df['min']) - (-df['min']).mean()) / ((-df['min']).std(ddof=0) if (-df['min']).std(ddof=0) != 0 else 1)
    df['severity_z'] = df[['wind_z', 'min_inv_z', 'category']].apply(lambda row: np.nanmean([row['wind_z'], row['min_inv_z'], (row['category'] - df['category'].mean()) / (df['category'].std(ddof=0) if df['category'].std(ddof=0) != 0 else 1)]), axis=1)

    # Final column list used in the model (kept in the dataframe)
    # ['log_alldeaths', 'masfem_c', 'wind', 'min', 'category', 'elapsedyrs', 'gender_mf', 'severity_z']

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> float:
    # Model: OLS predicting log deaths from centered name femininity controlling for storm intensity and time
    # Require that transform(df) has been run so masfem_c and log_alldeaths exist
    import statsmodels.api as sm

    df_model = df.dropna(subset=['log_alldeaths', 'masfem_c', 'wind', 'min', 'category', 'elapsedyrs'])

    # Predictors: masfem_c (IV) + severity controls. Use severity components as separate controls for transparency.
    X = df_model[['masfem_c', 'wind', 'min', 'category', 'elapsedyrs']].astype(float)
    X = sm.add_constant(X)
    y = df_model['log_alldeaths'].astype(float)

    ols_res = sm.OLS(y, X).fit()

    # Key summary metric: estimated coefficient on masfem_c (effect of a one-unit increase in name femininity on log deaths)
    coef_masfem = float(ols_res.params.get('masfem_c', np.nan))

    # Optionally, one might return p-value or standardized effect. Here we return the raw coefficient as the primary summary.
    return coef_masfem


