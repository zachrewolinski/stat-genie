from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a dataframe ready for modeling.

    Creates the following final columns used in the model:
      - NameFemininity: continuous femininity score (from original 'name').
      - IsFemaleName: binary indicator (1 if original 'elapsedyrs'==1, else 0).
      - Deaths: integer fatalities count from original 'ndam15'.
      - LogDeaths: log1p(Deaths) for sensitivity analyses.
      - LogDamage: log1p(ind) where 'ind' is normalized damages.
      - MaxWind: copied from 'wind'.
      - MinPressure: copied from 'min'.
      - SaffirSimpson: copied from 'masfem' (storm category).
      - Year: copied from 'alldeaths' (year of storm in this dataset).

    The function drops rows missing any required variables for the main analyses.
    """

    # Make a copy to avoid modifying in-place
    df = df.copy()

    # Rename / coerce key columns and create final columns
    # NameFemininity: original 'name' column (continuous masculinity-femininity index)
    if 'name' not in df.columns:
        raise KeyError("Input dataframe must contain a 'name' column with the masculinity/femininity index.")
    df['NameFemininity'] = pd.to_numeric(df['name'], errors='coerce')

    # IsFemaleName: original 'elapsedyrs' is a binary gender indicator (0 male, 1 female)
    if 'elapsedyrs' not in df.columns:
        raise KeyError("Input dataframe must contain an 'elapsedyrs' column with binary gender coding (0 male, 1 female).)")
    df['IsFemaleName'] = pd.to_numeric(df['elapsedyrs'], errors='coerce').fillna(0).astype(int)

    # Deaths: original 'ndam15' (total number of deaths)
    if 'ndam15' not in df.columns:
        raise KeyError("Input dataframe must contain an 'ndam15' column for fatalities.)")
    df['Deaths'] = pd.to_numeric(df['ndam15'], errors='coerce').fillna(0).astype(int)
    df['LogDeaths'] = np.log1p(df['Deaths'])

    # LogDamage: normalized property damage in 'ind' (use log1p to reduce skew)
    if 'ind' in df.columns:
        df['LogDamage'] = np.log1p(pd.to_numeric(df['ind'], errors='coerce').fillna(0))
    else:
        # create column of zeros if damage not present
        df['LogDamage'] = 0.0

    # MaxWind, MinPressure, SaffirSimpson, Year
    df['MaxWind'] = pd.to_numeric(df['wind'], errors='coerce') if 'wind' in df.columns else np.nan
    df['MinPressure'] = pd.to_numeric(df['min'], errors='coerce') if 'min' in df.columns else np.nan
    df['SaffirSimpson'] = pd.to_numeric(df['masfem'], errors='coerce') if 'masfem' in df.columns else np.nan

    # In this dataset 'alldeaths' was used to store the calendar year (per schema); fall back to 'year' if present
    if 'alldeaths' in df.columns:
        df['Year'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    elif 'year' in df.columns:
        df['Year'] = pd.to_numeric(df['year'], errors='coerce')
    else:
        df['Year'] = np.nan

    # Keep only rows with non-missing values in the variables needed for the primary model
    required_cols = ['NameFemininity', 'IsFemaleName', 'Deaths', 'MaxWind', 'MinPressure', 'SaffirSimpson', 'Year', 'LogDamage']
    df = df.dropna(subset=required_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit the primary statistical model and a sensitivity model.

    Primary model: Negative binomial GLM predicting Deaths from name femininity
    (continuous) and a binary female-name indicator, controlling for storm intensity
    and year. Negative binomial is appropriate for count outcomes with overdispersion.

    Sensitivity model: OLS on log1p(Deaths) to provide an alternative estimation.

    Returns a dict with keys 'nb_model' and 'ols_log_model' containing fitted results objects.
    """

    # Ensure the dataframe contains the transformed columns
    needed = ['Deaths', 'NameFemininity', 'IsFemaleName', 'MaxWind', 'MinPressure', 'SaffirSimpson', 'Year', 'LogDamage', 'LogDeaths']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe is missing required columns for modeling: {missing}")

    # Prepare predictors and response
    predictors = ['NameFemininity', 'IsFemaleName', 'MaxWind', 'MinPressure', 'SaffirSimpson', 'Year', 'LogDamage']
    X = df[predictors]
    X = sm.add_constant(X)
    y = df['Deaths']

    # Fit Negative Binomial GLM (counts) - using GLM with NegativeBinomial family
    # Note: uses the default link for NegativeBinomial family (log link)
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback to statsmodels.discrete NegativeBinomial if GLM fails
        from statsmodels.discrete.discrete_model import NegativeBinomial as NB_discrete
        nb_model = NB_discrete(y, X).fit(disp=False)

    # Sensitivity: OLS on log1p(Deaths)
    ols_model = sm.OLS(df['LogDeaths'], X).fit()

    # Return the fitted model results so the caller can inspect summaries, params, etc.
    results = {
        'nb_model': nb_model,
        'ols_log_model': ols_model,
        'predictors': predictors
    }
    return results


