from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    for col in ['masfem', 'min', 'wind', 'alldeaths', 'year', 'category', 'gender_mf', 'ndam15']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep rows with necessary variables for primary analysis
    df = df.dropna(subset=['masfem', 'alldeaths'])

    # Alldeaths as non-negative integer count used as dependent variable
    # Replace negative or non-integer values conservatively by coercing to int after clipping at 0
    df['alldeaths_count'] = df['alldeaths'].fillna(0).clip(lower=0).round().astype(int)

    # Log outcome (robustness): log(1 + alldeaths)
    df['log_alldeaths'] = np.log1p(df['alldeaths_count'])

    # Standardize continuous predictors to aid interpretation
    if 'masfem' in df.columns:
        df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_z'] = np.nan

    if 'wind' in df.columns:
        df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1)
    else:
        df['wind_z'] = np.nan

    if 'min' in df.columns:
        df['min_z'] = (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1)
    else:
        df['min_z'] = np.nan

    # Center year to improve numerical stability
    if 'year' in df.columns:
        df['year_c'] = df['year'] - df['year'].mean()
    else:
        df['year_c'] = np.nan

    # Binary female-coded name indicator (0/1) from existing gender_mf (0=male,1=female)
    if 'gender_mf' in df.columns:
        df['gender_female'] = df['gender_mf'].astype(float)
    else:
        df['gender_female'] = np.nan

    # Ensure category is present and integer (treat as categorical in models)
    if 'category' in df.columns:
        df['category'] = pd.to_numeric(df['category'], errors='coerce').round().astype('Int64')

    # Source: ensure string dtype and fillna
    if 'source' in df.columns:
        df['source'] = df['source'].astype(str).fillna('unknown')
    else:
        df['source'] = 'unknown'

    # Final drop: drop any rows that still have missing values in the modeling columns
    required_cols = ['masfem_z', 'alldeaths_count', 'wind_z', 'min_z', 'category', 'year_c', 'gender_female', 'source']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Return transformed dataframe containing all columns used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Expectation: df is already transformed via transform(); uses columns specified in cvars

    # Primary model: Negative binomial regression for count outcome alldeaths_count
    # Model includes masfem (standardized) and key storm controls; categorical variables are handled via C(...)
    formula_nb = 'alldeaths_count ~ masfem_z + wind_z + min_z + C(category) + year_c + gender_female + C(source)'
    try:
        nb_model = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
        # Obtain robust (HC3) covariance estimates for inference
        nb_results = nb_model.get_robustcov_results(cov_type='HC3')
    except Exception as e:
        nb_results = e

    # Robustness check 1: OLS on log(1 + deaths)
    formula_ols = 'log_alldeaths ~ masfem_z + wind_z + min_z + C(category) + year_c + gender_female + C(source)'
    try:
        ols_model = smf.ols(formula_ols, data=df).fit(cov_type='HC3')
        ols_results = ols_model
    except Exception as e:
        ols_results = e

    # Robustness check 2: use binary gender-coded name as predictor instead of continuous masfem
    formula_nb_gender = 'alldeaths_count ~ gender_female + wind_z + min_z + C(category) + year_c + C(source)'
    try:
        nb_model_gender = smf.glm(formula_nb_gender, data=df, family=sm.families.NegativeBinomial()).fit()
        nb_gender_results = nb_model_gender.get_robustcov_results(cov_type='HC3')
    except Exception as e:
        nb_gender_results = e

    # Package results
    results = {
        'nb_main': nb_results,
        'ols_log_outcome': ols_results,
        'nb_gender_binary': nb_gender_results
    }

    return results


