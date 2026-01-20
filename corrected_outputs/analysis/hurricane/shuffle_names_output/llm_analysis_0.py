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
    Transforms the raw hurricane dataset into a dataframe with the exact columns
    used in the analysis models. The function:
      - copies the input dataframe
      - coerces relevant columns to numeric types
      - constructs the outcome (TotalDeaths and LogTotalDeaths)
      - constructs predictor (Femininity) and controls
      - drops rows with missing key IV or DV
      - returns only the columns used in models (plus reference columns)
    """
    df = df.copy()

    # Primary predictor: continuous masculinity-femininity index (higher = more feminine)
    # 'name' in this dataset is the masculinity-femininity index per documentation
    df['Femininity'] = pd.to_numeric(df.get('name'), errors='coerce')

    # Outcome: total deaths. 'ndam15' described as total number of deaths in the schema
    df['TotalDeaths'] = pd.to_numeric(df.get('ndam15'), errors='coerce')
    # Log-transform the deaths to reduce skew and make OLS assumptions more plausible
    df['LogTotalDeaths'] = np.log(df['TotalDeaths'].fillna(0) + 1)

    # Controls: intensity / exposure variables
    df['MaxWindMPH'] = pd.to_numeric(df.get('wind'), errors='coerce')
    # The schema text indicates 'gender_mf' holds min pressure values in hPa for many rows
    df['MinPressure'] = pd.to_numeric(df.get('gender_mf'), errors='coerce')
    df['SaffirSimpson'] = pd.to_numeric(df.get('masfem'), errors='coerce')
    df['Year'] = pd.to_numeric(df.get('alldeaths'), errors='coerce')
    df['Damage'] = pd.to_numeric(df.get('ind'), errors='coerce')

    # Historical binary name-gender indicator (0 = male, 1 = female) for robustness
    df['FemaleName'] = pd.to_numeric(df.get('elapsedyrs'), errors='coerce')

    # Keep original name string and source id for reference / merging if needed
    df['StormName'] = df.get('ndam')
    df['SourceID'] = df.get('source')

    # Drop rows missing the key predictor or outcome
    df = df.dropna(subset=['Femininity', 'TotalDeaths'])

    # Optionally, drop rows with all intensity controls missing (not strictly necessary)
    # but keep as is to maximize sample; models will handle NAs by error if present.

    # Return only the columns we will use in modeling (plus references).
    return df[[
        'Femininity', 'TotalDeaths', 'LogTotalDeaths',
        'FemaleName', 'MaxWindMPH', 'MinPressure', 'SaffirSimpson', 'Year', 'Damage',
        'StormName', 'SourceID'
    ]]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs the main statistical analyses testing whether more feminine hurricane names
    are associated with higher fatalities (proxy for fewer precautions).

    Primary model
      - OLS regression of LogTotalDeaths on Femininity and controls, with robust SEs (HC3).

    Robustness model
      - Negative binomial GLM on the count TotalDeaths (to account for count nature and overdispersion).

    Returns a dict with the fitted results objects so user can inspect summaries.
    """
    df = df.copy()

    # Define predictor matrix and outcome for OLS
    X_cols = ['Femininity', 'FemaleName', 'MaxWindMPH', 'MinPressure', 'SaffirSimpson', 'Year', 'Damage']
    # Ensure columns exist; if some are missing (all-NA) raise informative error
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['LogTotalDeaths'].astype(float)

    # Fit OLS with robust standard errors (HC3)
    ols_model = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

    # Robustness: Negative Binomial on the raw counts
    # Use the same covariates (without log transform) for a GLM count model
    nb_X = sm.add_constant(df[X_cols].astype(float), has_constant='add')
    nb_y = df['TotalDeaths'].astype(float)
    # Use GLM NegativeBinomial family (handles overdispersion better than Poisson)
    try:
        nb_model = sm.GLM(nb_y, nb_X, family=sm.families.NegativeBinomial(), missing='drop').fit()
    except Exception as e:
        nb_model = None
        print('Negative binomial model failed:', e)

    # Return models in a dict; user can call .summary() on each if not None
    results = {
        'ols_log_deaths': ols_model,
        'neg_binom_counts': nb_model
    }
    return results


