from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (1978) survey dataframe into the analysis dataframe.

    Output columns (exact names used in the model):
      - Affairs: integer count of extramarital affairs
      - Children: binary indicator 1 if children present, 0 otherwise
      - Female: binary indicator 1 if respondent is female, 0 if male
      - Age_c: age centered around sample mean
      - Yearsmarried_c: yearsmarried centered around sample mean
      - religiousness, education, occupation, rating: numeric controls

    The function drops rows with missing values in the variables required for the model.
    """
    df = df.copy()
    # Columns required for analysis
    required_cols = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Dependent variable: integer count from 'affairs'
    # The original encoding uses 0,1,2,3,7,12 etc.; keep numeric as-is
    df['Affairs'] = pd.to_numeric(df['affairs'], errors='coerce').astype(int)

    # Independent variable: children present
    # Original values are 'yes'/'no' (factor). Map to 1/0.
    df['Children'] = df['children'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Gender -> Female binary
    df['Female'] = (df['gender'].astype(str).str.lower() == 'female').astype(int)

    # Ensure numeric types for controls and create centered versions where noted
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['yearsmarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['education'] = pd.to_numeric(df['education'], errors='coerce')
    df['occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['Affairs', 'Children', 'Female', 'age', 'yearsmarried',
                           'religiousness', 'education', 'occupation', 'rating'])

    # Center age and years married to improve interpretability / numerical stability
    df['Age_c'] = df['age'] - df['age'].mean()
    df['Yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()

    # Keep only columns required by the model (exact column names used below)
    out_cols = ['Affairs', 'Children', 'Female', 'Age_c', 'Yearsmarried_c',
                'religiousness', 'education', 'occupation', 'rating']
    return df[out_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a zero-inflated negative binomial (ZINB) model for count data with excess zeros.

    Rationale:
      - 'Affairs' is a count variable with many zeros (no extramarital sex); ZINB models a count process
        and a separate logit process for excess zeros (structural zeros).
      - The primary parameter of interest is the coefficient on 'Children' in the count equation
        (and optionally in the inflation equation). We include the same covariates in both parts.

    Returns the fitted statsmodels results object (so the caller can examine params, SEs, CI, etc.).
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Ensure required columns exist
    exog_cols = ['Children', 'Female', 'Age_c', 'Yearsmarried_c',
                 'religiousness', 'education', 'occupation', 'rating']

    # Prepare exogenous variables (add constant)
    exog = sm.add_constant(df[exog_cols], has_constant='add')
    endog = df['Affairs']

    # Fit Zero-Inflated Negative Binomial (logit inflation). Use same covariates for both parts.
    zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog, inflation='logit')

    # Fit with a robust optimizer and increased iterations if needed
    try:
        res = zinb.fit(method='bfgs', maxiter=200, disp=False)
    except Exception:
        # Fall back to default fit if bfgs fails
        res = zinb.fit(disp=False)

    # Return the full results object for inspection (params, pvalues, summary, etc.)
    return res


