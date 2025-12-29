from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/affairs/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling. Returns a dataframe containing the exact columns named
    in the conceptual variables section:
      - affairs_count
      - log_affairs_plus1
      - children_yes
      - gender_male
      - age
      - yearsmarried
      - religiousness
      - education
      - occupation
      - rating

    The function cleans missing values for these columns and coerces types.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure affairs is numeric and rename as affairs_count
    # If there are string encodings or NaNs, coerce to numeric
    df['affairs_count'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Create logged outcome for OLS robustness (log(affairs + 1))
    df['log_affairs_plus1'] = np.log1p(df['affairs_count'])

    # Standardize and create children indicator (1 if 'yes', 0 if 'no')
    # handle possible capitalization or whitespace
    if df['children'].dtype == object or pd.api.types.is_categorical_dtype(df['children']):
        children_series = df['children'].astype(str).str.strip().str.lower()
        df['children_yes'] = np.where(children_series == 'yes', 1,
                                      np.where(children_series == 'no', 0, np.nan))
    else:
        # If encoded differently (e.g., 1/0), attempt numeric coercion
        df['children_yes'] = pd.to_numeric(df['children'], errors='coerce')

    # Gender dummy: 1 for male, 0 for female
    if df['gender'].dtype == object or pd.api.types.is_categorical_dtype(df['gender']):
        gender_series = df['gender'].astype(str).str.strip().str.lower()
        df['gender_male'] = np.where(gender_series == 'male', 1,
                                     np.where(gender_series == 'female', 0, np.nan))
    else:
        df['gender_male'] = pd.to_numeric(df['gender'], errors='coerce')

    # Ensure numeric columns are numeric; coerce errors to NaN
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If any expected control is missing entirely, create column of NaNs (model will drop rows)
            df[col] = np.nan

    # Keep only rows that have non-missing values in the dependent variable and primary IV and at least some controls
    required_cols = ['affairs_count', 'children_yes', 'gender_male', 'age', 'yearsmarried',
                     'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_cols)

    # Cast integer-like columns to integers where appropriate
    df['affairs_count'] = df['affairs_count'].astype(int)
    df['children_yes'] = df['children_yes'].astype(int)
    df['gender_male'] = df['gender_male'].astype(int)

    # Final column order (keeps only the columns needed for modeling)
    df = df[[
        'affairs_count',
        'log_affairs_plus1',
        'children_yes',
        'gender_male',
        'age',
        'yearsmarried',
        'religiousness',
        'education',
        'occupation',
        'rating'
    ]].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit several models to examine the association between having children and extramarital affairs:
      1) OLS on log(affairs + 1) with robust standard errors (interpretation: % differences approx),
      2) Poisson GLM on the raw affairs count,
      3) Negative Binomial GLM on the raw affairs count (to allow overdispersion).

    Returns a dict with fitted results objects for each model.
    """
    results = {}

    # Ensure required columns present
    required = ['affairs_count', 'log_affairs_plus1', 'children_yes', 'gender_male', 'age',
                'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Predictor list
    predictors = ['children_yes', 'gender_male', 'age', 'yearsmarried',
                  'religiousness', 'education', 'occupation', 'rating']

    X = sm.add_constant(df[predictors])

    # 1) OLS on log(affairs + 1)
    ols_model = sm.OLS(df['log_affairs_plus1'], X).fit(cov_type='HC3')  # robust SEs
    results['ols'] = ols_model

    # 2) Poisson GLM on count
    poisson_model = sm.GLM(df['affairs_count'], X, family=sm.families.Poisson()).fit()
    results['poisson'] = poisson_model

    # 3) Negative Binomial to account for over-dispersion
    try:
        nb_family = sm.families.NegativeBinomial()
        nb_model = sm.GLM(df['affairs_count'], X, family=nb_family).fit()
        results['neg_binomial'] = nb_model
    except Exception:
        # If GLM NegativeBinomial fails (older statsmodels), fallback to discrete model
        try:
            from statsmodels.discrete.discrete_model import NegativeBinomial as NB_discrete
            nb_discrete = NB_discrete(df['affairs_count'], X).fit(disp=False)
            results['neg_binomial'] = nb_discrete
        except Exception as e:
            results['neg_binomial_error'] = str(e)

    # Return fitted result objects so the caller can inspect summaries, coefficients, and diagnostics
    return results


