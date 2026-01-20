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
    # Work on a copy
    df = df.copy()

    # --- Map / coerce key columns to numeric and create clear column names ---
    # 'name' is the continuous masculinity-femininity score (higher = more feminine)
    df['NameFemScore'] = pd.to_numeric(df.get('name'), errors='coerce')

    # 'elapsedyrs' in this dataset acts as a binary gender indicator (0=male, 1=female)
    df['FemaleName'] = pd.to_numeric(df.get('elapsedyrs'), errors='coerce')

    # Deaths (raw counts) - use 'ndam15' which is described as total deaths
    df['Deaths'] = pd.to_numeric(df.get('ndam15'), errors='coerce')

    # Objective storm severity measures
    df['Wind'] = pd.to_numeric(df.get('wind'), errors='coerce')
    df['PressureMin'] = pd.to_numeric(df.get('min'), errors='coerce')

    # Economic damage index (normalized property damage) - column 'ind'
    df['DamageIndex'] = pd.to_numeric(df.get('ind'), errors='coerce')

    # Year of storm - dataset column named 'alldeaths' contains the year in this file
    df['Year'] = pd.to_numeric(df.get('alldeaths'), errors='coerce')

    # Storm category (if present) - 'masfem' in this file appears to encode category-like values
    df['StormCat'] = pd.to_numeric(df.get('masfem'), errors='coerce')

    # --- Derived variables ---
    # Log transform of deaths for OLS modeling
    df['LogDeaths'] = np.log1p(df['Deaths'])

    # Log transform of damage to reduce skew
    df['LogDamage'] = np.log1p(df['DamageIndex'])

    # --- Drop rows missing essential variables ---
    # We need at minimum the name score and deaths and at least one severity metric.
    df = df.dropna(subset=['NameFemScore', 'Deaths', 'Wind', 'PressureMin'])

    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['LogDeaths'])

    # --- Center continuous predictors used in the model to improve interpretability and reduce collinearity ---
    # We create centered versions that will be used by the statistical model.
    cont_to_center = ['NameFemScore', 'Wind', 'PressureMin', 'LogDamage', 'Year', 'StormCat']
    for col in cont_to_center:
        if col in df.columns:
            # compute mean/std using available observations; if std==0, leave as zero-centered
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            if pd.isna(std) or std == 0:
                df[col + '_c'] = df[col] - mean
            else:
                df[col + '_c'] = (df[col] - mean) / std

    # Final check: drop rows with any remaining NaNs in the model columns
    model_cols = ['NameFemScore_c', 'FemaleName', 'Wind_c', 'PressureMin_c', 'LogDamage_c', 'Year_c', 'StormCat_c', 'LogDeaths', 'Deaths']
    # keep only those model columns that exist in the dataframe
    model_cols = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=model_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # Work on a copy
    df = df.copy()

    # Define predictors as used in the conceptual model (centered continuous variables + binary female indicator)
    predictors = ['NameFemScore_c', 'FemaleName', 'Wind_c', 'PressureMin_c', 'LogDamage_c', 'Year_c', 'StormCat_c']
    # Keep only predictors that exist in the transformed dataframe
    predictors = [p for p in predictors if p in df.columns]

    # Build design matrix
    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')

    # Endogenous variables: raw counts for count model, log-deaths for OLS
    y_counts = df['Deaths']
    y_log = df['LogDeaths']

    results = {}

    # 1) Primary count model: Negative Binomial (models counts directly, accounts for over-dispersion)
    # If NegativeBinomial family is not available for GLM in the present statsmodels version, fall back to Poisson with robust SE.
    try:
        nb = sm.GLM(y_counts, X, family=sm.families.NegativeBinomial())
        nb_res = nb.fit()
        results['nb_model'] = nb_res
    except Exception as e:
        # fallback: Poisson with robust covariance (heteroskedasticity-consistent)
        pois = sm.GLM(y_counts, X, family=sm.families.Poisson())
        pois_res = pois.fit()  # we'll get robust cov below
        pois_res_robust = pois_res.get_robustcov_results(cov_type='HC3')
        results['poisson_robust'] = pois_res_robust

    # 2) OLS on log-transformed deaths as a complementary specification (interpretable coefficients)
    ols = sm.OLS(y_log, X)
    ols_res = ols.fit()
    # Provide robust standard errors for inference
    ols_res_robust = ols_res.get_robustcov_results(cov_type='HC3')
    results['ols_log_deaths_robust'] = ols_res_robust

    # Return the fitted result objects (callers can inspect summary(), params, conf_int(), etc.)
    return results


