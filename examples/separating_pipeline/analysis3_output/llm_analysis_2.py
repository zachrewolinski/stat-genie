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
    """
    Transform the raw hurricane dataset into a dataframe ready for modeling.

    Outputs (kept/created columns):
      - FemininityIndex: continuous index (z-scored name ratings and MTurk ratings averaged when available)
      - LogDeaths: log(ndam15 + 1)
      - StormIntensity: composite intensity = z(wind) - z(min pressure)
      - MaxWind: original 'wind'
      - MinPressure: original 'min'
      - Category: numeric Saffir-Simpson category from 'masfem'
      - LogDamage: log(ind + 1)
      - Year: year of event from 'alldeaths'
      - FemaleBinary: binary indicator from 'elapsedyrs' when present
    """
    df = df.copy()

    # Ensure numeric conversions for key columns
    numeric_cols = ['ndam15', 'name', 'wind', 'min', 'masfem', 'ind', 'alldeaths', 'elapsedyrs']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create FemininityIndex using available femininity measures
    fem_cols = []
    if 'name' in df.columns:
        fem_cols.append('name')
    if 'masfem_mturk' in df.columns:
        # ensure numeric conversion if present
        df['masfem_mturk'] = pd.to_numeric(df['masfem_mturk'], errors='coerce')
        fem_cols.append('masfem_mturk')

    if len(fem_cols) == 0:
        # no femininity measures found: create column with NaNs
        df['FemininityIndex'] = np.nan
    else:
        # z-score each available femininity column (using population std, ddof=0)
        zcols = []
        for c in fem_cols:
            zcol = c + '_z'
            # avoid division by zero if std==0
            std = df[c].std(ddof=0)
            if pd.isna(std) or std == 0:
                df[zcol] = (df[c] - df[c].mean())
            else:
                df[zcol] = (df[c] - df[c].mean()) / std
            zcols.append(zcol)
        # average z-scored femininity measures row-wise (ignores NaNs)
        df['FemininityIndex'] = df[zcols].mean(axis=1)

    # Outcome: log fatalities (proxy for insufficient precautions / consequences)
    if 'ndam15' in df.columns:
        df['LogDeaths'] = np.log(df['ndam15'].clip(lower=0) + 1)
    else:
        df['LogDeaths'] = np.nan

    # Controls: intensity measures
    if 'wind' in df.columns:
        df['MaxWind'] = df['wind']
        wind_std = df['MaxWind'].std(ddof=0)
        if pd.isna(wind_std) or wind_std == 0:
            df['Wind_z'] = df['MaxWind'] - df['MaxWind'].mean()
        else:
            df['Wind_z'] = (df['MaxWind'] - df['MaxWind'].mean()) / wind_std
    else:
        df['MaxWind'] = np.nan
        df['Wind_z'] = np.nan

    if 'min' in df.columns:
        df['MinPressure'] = df['min']
        min_std = df['MinPressure'].std(ddof=0)
        if pd.isna(min_std) or min_std == 0:
            df['MinPressure_z'] = df['MinPressure'] - df['MinPressure'].mean()
        else:
            df['MinPressure_z'] = (df['MinPressure'] - df['MinPressure'].mean()) / min_std
    else:
        df['MinPressure'] = np.nan
        df['MinPressure_z'] = np.nan

    # Composite StormIntensity: higher = more intense
    df['StormIntensity'] = df['Wind_z'] - df['MinPressure_z']

    # Category (Saffir-Simpson) numeric control
    if 'masfem' in df.columns:
        df['Category'] = pd.to_numeric(df['masfem'], errors='coerce')
    else:
        df['Category'] = np.nan

    # Economic damage control (log-transformed)
    if 'ind' in df.columns:
        # 'ind' described as normalized economic damage
        df['LogDamage'] = np.log(df['ind'].clip(lower=0) + 1)
    else:
        df['LogDamage'] = np.nan

    # Year control
    if 'alldeaths' in df.columns:
        df['Year'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    else:
        df['Year'] = np.nan

    # Binary female name indicator when available
    if 'elapsedyrs' in df.columns:
        # elapsedyrs described as binary gender indicator of the hurricane name (0 male, 1 female)
        df['FemaleBinary'] = df['elapsedyrs'].apply(lambda x: int(x) if pd.notnull(x) else np.nan)
    else:
        df['FemaleBinary'] = np.nan

    # Drop rows missing the outcome or main IV
    df = df.dropna(subset=['LogDeaths', 'FemininityIndex'])

    # Keep only the columns necessary for modeling + some diagnostics
    keep_cols = [
        'FemininityIndex', 'LogDeaths', 'StormIntensity', 'MaxWind', 'MinPressure',
        'Category', 'LogDamage', 'Year', 'FemaleBinary',
        # keep raw originals for reference if present
        'ndam15', 'name', 'masfem_mturk', 'wind', 'min', 'ind', 'alldeaths'
    ]
    # Only keep columns that exist in df
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit an OLS model predicting LogDeaths from FemininityIndex while controlling for storm
    intensity, category, year, and damage. Returns the fitted statsmodels results object with robust SEs.

    Primary specification:
      LogDeaths ~ FemininityIndex + StormIntensity + Category + Year + LogDamage + MaxWind + MinPressure

    Additional: if FemaleBinary is present it is included as an extra control.
    Standard errors: heteroskedasticity-robust (HC3).
    """
    df = df.copy()

    # Define outcome and predictors
    y = df['LogDeaths']

    predictors = ['FemininityIndex', 'StormIntensity', 'Category', 'Year', 'LogDamage', 'MaxWind', 'MinPressure']
    # keep only predictors present in df
    predictors = [p for p in predictors if p in df.columns]

    # Optionally add FemaleBinary as a control if available
    if 'FemaleBinary' in df.columns:
        predictors.append('FemaleBinary')

    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust standard errors (HC3)
    model_res = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

    # For convenience, attach the design info used
    model_res.model_data = {
        'n_obs': int(model_res.nobs),
        'predictors': predictors
    }

    return model_res


