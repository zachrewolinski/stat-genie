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
    Transform the raw Fair affairs dataset into a modeling-ready dataframe.

    Outputs (columns used in models):
      - AffairsCount: numeric count of affairs (from 'affairs')
      - HasChildren: binary 1=yes 0=no (from 'children')
      - IsFemale: binary 1=female 0=male (from 'gender')
      - age_z, yearsmarried_z, religiousness_z, education_z, occupation_z, rating_z: standardized controls
      - LogAffairsPlus1: log(1 + AffairsCount) for an alternative OLS specification

    The function drops rows with missing required fields.
    """
    df = df.copy()

    # Required raw columns: 'affairs', 'children', 'gender', 'age', 'yearsmarried',
    # 'religiousness', 'education', 'occupation', 'rating'

    # Convert affairs to numeric and drop missing
    df['AffairsCount'] = pd.to_numeric(df['affairs'], errors='coerce')
    df = df.dropna(subset=['AffairsCount'])

    # Map children to binary: 'yes' -> 1, 'no' -> 0 (case-insensitive). Drop if mapping fails.
    df['children_str'] = df['children'].astype(str).str.strip().str.lower()
    df['HasChildren'] = df['children_str'].map({'yes': 1, 'no': 0})
    df = df.drop(columns=['children_str'])
    df = df.dropna(subset=['HasChildren'])
    df['HasChildren'] = df['HasChildren'].astype(int)

    # Map gender to binary IsFemale: 'female' -> 1, 'male' -> 0. If other, set to NaN and drop.
    df['gender_str'] = df['gender'].astype(str).str.strip().str.lower()
    df['IsFemale'] = df['gender_str'].map({'female': 1, 'male': 0})
    df = df.drop(columns=['gender_str'])
    df = df.dropna(subset=['IsFemale'])
    df['IsFemale'] = df['IsFemale'].astype(int)

    # Coerce numeric controls to numeric and drop rows with missing controls (these are important confounders)
    numeric_controls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_controls:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_controls)

    # Standardize controls (z-score) for stable coefficients
    for col in numeric_controls:
        zcol = f"{col}_z"
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # If std is zero (unlikely), set z to zero to avoid division by zero
        if std == 0 or np.isnan(std):
            df[zcol] = 0.0
        else:
            df[zcol] = (df[col] - mean) / std

    # Log transform of affairs for an auxiliary OLS model
    df['LogAffairsPlus1'] = np.log1p(df['AffairsCount'].astype(float))

    # Final check: keep only columns necessary for modeling and return full dataframe copy
    # (We keep originals also; the model code will select the exact columns it needs.)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary specifications to estimate the effect of having children on extramarital affairs:
      1) Negative binomial GLM for count outcome (primary model)
      2) OLS on log(1 + affairs) as a robustness check

    Returns a dict with model fit objects:
      - 'neg_binom': fitted statsmodels GLMResults for Negative Binomial
      - 'ols_log': fitted statsmodels OLSResults for log(affairs+1)
    """
    import statsmodels.api as sm

    # Required transformed columns (as produced by transform):
    exog_vars = [
        'HasChildren',
        'IsFemale',
        'age_z',
        'yearsmarried_z',
        'religiousness_z',
        'education_z',
        'occupation_z',
        'rating_z'
    ]

    # Ensure the columns exist
    missing = [c for c in exog_vars + ['AffairsCount', 'LogAffairsPlus1'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Prepare design matrix
    X = df[exog_vars].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y_count = df['AffairsCount'].astype(float)

    # 1) Negative binomial GLM for counts (robust choice for overdispersed counts)
    try:
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If NB fails, return the exception information in place of the fit
        nb_model = e

    # 2) OLS on log(affairs + 1) as a robustness check
    y_log = df['LogAffairsPlus1'].astype(float)
    ols_model = sm.OLS(y_log, X).fit()

    # Pack results
    results = {
        'neg_binom': nb_model,
        'ols_log': ols_model,
        'exog_vars': exog_vars
    }

    return results


