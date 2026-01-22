from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/noperturb_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (Psychology Today) dataset to produce analysis-ready columns.

    Outputs (columns created or required by downstream models):
      - has_children: binary 1/0
      - is_female: binary 1/0
      - age_c, yearsmarried_c, religiousness_c, education_c, occupation_c, rating_c: centered numeric controls
      - affairs: numeric dependent variable (kept from input)
      - any_affair: binary indicator (affairs > 0) for logistic robustness

    The function:
      - drops rows missing any required variables
      - normalizes/casts types
      - centers continuous controls (mean 0)
    """
    df = df.copy()

    # Required raw columns
    required = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    # Drop rows missing any required column
    df = df.dropna(subset=required)

    # Normalize and map 'children' to binary has_children
    # Accept common textual encodings; fallback: drop unmapped
    df['children'] = df['children'].astype(str).str.strip().str.lower()
    df['has_children'] = df['children'].map({
        'yes': 1, 'y': 1, 'true': 1, '1': 1,
        'no': 0, 'n': 0, 'false': 0, '0': 0
    })
    # If mapping left NaN (unexpected encoding) attempt exact match for 'yes'/'no' capitalizations
    df.loc[df['children'] == 'yes', 'has_children'] = 1
    df.loc[df['children'] == 'no', 'has_children'] = 0

    # Drop rows where has_children still missing
    df = df.dropna(subset=['has_children'])
    df['has_children'] = df['has_children'].astype(int)

    # Map gender to is_female binary
    df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    df['is_female'] = df['gender'].map({'female': 1, 'f': 1, 'male': 0, 'm': 0})
    df = df.dropna(subset=['is_female'])
    df['is_female'] = df['is_female'].astype(int)

    # Ensure numeric columns are numeric; coerce invalids to NaN then drop
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'affairs']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with numeric conversion failures for essential variables
    df = df.dropna(subset=numeric_cols)

    # Keep affairs as-is (numeric count/top-coded). Create a binary any_affair for logistic model
    df['any_affair'] = (df['affairs'] > 0).astype(int)

    # Center continuous controls for better numerical behavior and interpretation
    df['age_c'] = df['age'] - df['age'].mean()
    df['yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()
    df['religiousness_c'] = df['religiousness'] - df['religiousness'].mean()
    df['education_c'] = df['education'] - df['education'].mean()
    df['occupation_c'] = df['occupation'] - df['occupation'].mean()
    df['rating_c'] = df['rating'] - df['rating'].mean()

    # Final check: ensure no missing values in model columns
    model_cols = ['affairs', 'has_children', 'is_female', 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c', 'any_affair']
    df = df.dropna(subset=model_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit multiple models to assess whether having children decreases reported engagement in extramarital affairs.

    Models estimated:
      1) OLS (robust SEs) on raw affairs counts (linear approximation)
      2) Negative Binomial GLM on affairs counts (accounts for over-dispersion relative to Poisson)
      3) Logistic regression on any_affair (binary outcome: any vs none) as a robustness check

    Returns a dictionary with fitted model result objects keyed by 'ols', 'neg_bin', and 'logit'.
    """
    df = df.copy()

    # Design matrix: same controls used across models
    feature_cols = ['has_children', 'is_female', 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c']
    X = df[feature_cols]
    X = sm.add_constant(X)

    # Dependent variables
    y_count = df['affairs']
    y_binary = df['any_affair']

    results = {}

    # 1) OLS with robust standard errors (HC3)
    ols_mod = sm.OLS(y_count, X)
    ols_res = ols_mod.fit(cov_type='HC3')
    results['ols'] = ols_res

    # 2) Negative Binomial (GLM) for count outcome
    # Use GLM NegativeBinomial family to allow overdispersion relative to Poisson
    nb_mod = sm.GLM(y_count, X, family=sm.families.NegativeBinomial())
    nb_res = nb_mod.fit()
    results['neg_bin'] = nb_res

    # 3) Logistic regression on any_affair (robustness check)
    logit_mod = sm.Logit(y_binary, X)
    # suppress full solver output for cleaner runs
    try:
        logit_res = logit_mod.fit(disp=False)
    except Exception:
        # fallback to default fit with printing if an optimization warning arises
        logit_res = logit_mod.fit()
    results['logit'] = logit_res

    # Optionally print summaries for quick inspection (comment/uncomment as needed)
    # print('\nOLS results:\n', results['ols'].summary())
    # print('\nNegative Binomial results:\n', results['neg_bin'].summary())
    # print('\nLogit results:\n', results['logit'].summary())

    return results


