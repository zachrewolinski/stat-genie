from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a modeling dataframe containing:
      - deaths: raw death counts (from 'ndam15')
      - log_deaths: log(1 + deaths) for robustness/OLS models
      - femininity: original continuous masculinity-femininity index from 'name' (higher = more feminine)
      - female_name: binary indicator from 'elapsedyrs' (0 male, 1 female)
      - standardized controls: wind_z, min_z, ind_z, year_z
    The function drops rows missing essential variables.
    """
    # Ensure we operate on a copy
    df = df.copy()

    # Normalize/clean column names expected in the dataset
    # The dataset schema indicates:
    #  - 'name' is the continuous masculinity-femininity index (higher -> more feminine)
    #  - 'ndam15' is total number of deaths
    #  - 'elapsedyrs' is a binary indicator (0 male, 1 female)
    #  - 'wind', 'min', 'ind', 'alldeaths' provide intensity/damage/year info

    # Convert relevant columns to numeric, coercing errors to NaN
    numeric_cols = ['name', 'ndam15', 'elapsedyrs', 'wind', 'min', 'ind', 'alldeaths']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Rename/derive columns used in analysis
    # deaths: use ndam15 (total deaths). If ndam15 not present but ndam exists, user can modify accordingly.
    if 'ndam15' in df.columns:
        df['deaths'] = df['ndam15']
    elif 'ndam' in df.columns:
        # fallback: try 'ndam' if present and numeric
        df['deaths'] = pd.to_numeric(df['ndam'], errors='coerce')
    else:
        df['deaths'] = np.nan

    # femininity continuous score
    df['femininity'] = df['name'] if 'name' in df.columns else np.nan

    # binary female name indicator
    # elapsedyrs is described as 0 for male, 1 for female in the schema; coerce to integer 0/1
    if 'elapsedyrs' in df.columns:
        df['female_name'] = df['elapsedyrs'].fillna(0).astype(int)
    else:
        df['female_name'] = 0

    # Year: schema uses 'alldeaths' as year
    if 'alldeaths' in df.columns:
        df['year'] = df['alldeaths']
    elif 'year' in df.columns:
        df['year'] = df['year']
    else:
        df['year'] = np.nan

    # Controls: wind, min (pressure), ind (normalized damage)
    df['wind'] = df['wind'] if 'wind' in df.columns else np.nan
    df['min'] = df['min'] if 'min' in df.columns else np.nan
    df['ind'] = df['ind'] if 'ind' in df.columns else np.nan

    # Drop rows missing the primary outcome or primary IV
    df = df.dropna(subset=['deaths', 'femininity'])

    # Create a logged death outcome for OLS robustness
    df['log_deaths'] = np.log1p(df['deaths'])

    # Standardize continuous predictors for interpretability in the model
    def zscore(series: pd.Series) -> pd.Series:
        if series.dropna().shape[0] <= 1:
            return (series - series.mean())
        return (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1.0)

    # Standardize femininity and controls
    df['fem_z'] = zscore(df['femininity'])
    df['wind_z'] = zscore(df['wind'])
    df['min_z'] = zscore(df['min'])
    df['ind_z'] = zscore(df['ind'])
    df['year_z'] = zscore(df['year'])

    # Keep only columns necessary for modeling (but return full df copy with these columns present)
    # The model function will select the columns it needs by name.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a primary negative binomial regression of death counts on name femininity
    controlling for storm intensity and damage, and a robustness OLS on log_deaths.

    Returns a dictionary with fitted model results objects (statsmodels results).
    """
    # Columns expected in df (output of transform):
    # 'deaths' (count), 'log_deaths', 'fem_z', 'female_name', 'wind_z', 'min_z', 'ind_z', 'year_z'

    # Select model predictors
    exog_cols = ['fem_z', 'female_name', 'wind_z', 'min_z', 'ind_z', 'year_z']
    missing = [c for c in exog_cols + ['deaths', 'log_deaths'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop rows with any NA in the predictors or outcomes used for the model
    model_df = df.dropna(subset=exog_cols + ['deaths', 'log_deaths']).copy()

    # Add constant
    X = sm.add_constant(model_df[exog_cols])
    y_counts = model_df['deaths']

    # Primary model: Negative Binomial GLM (handles overdispersed count data)
    try:
        nb_model = sm.GLM(y_counts, X, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback to Poisson if NB fails to converge
        nb_model = sm.GLM(y_counts, X, family=sm.families.Poisson()).fit()

    # Robustness: OLS on log(1 + deaths)
    ols_model = sm.OLS(model_df['log_deaths'], X).fit(cov_type='HC3')

    # Return results objects; callers can inspect .summary()
    return {
        'nb_model': nb_model,
        'ols_log_model': ols_model,
        'model_dataframe': model_df
    }


