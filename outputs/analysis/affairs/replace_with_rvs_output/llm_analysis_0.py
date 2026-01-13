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
    Transform the raw Fair (1978) affairs dataset for modeling.

    Outputs a dataframe containing the following columns used in the model:
      - affairs: integer count outcome (keeps original column name)
      - Children: binary (1=yes, 0=no)
      - GenderMale: binary (1=male, 0=female)
      - Children_x_Male: interaction Children * GenderMale (for moderation)
      - age_c, yearsmarried_c, religiousness_c, education_c, occupation_c, rating_c: mean-centered numeric controls

    Steps:
      - Drop rows missing key columns
      - Map categorical flags to binary
      - Ensure numeric columns are numeric
      - Mean-center continuous controls
    """
    df = df.copy()

    # Ensure required columns exist; if not, this will raise a KeyError so the user can inspect the dataset
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Drop rows missing the outcome or main IV or moderator
    df = df.dropna(subset=['affairs', 'children', 'gender'])

    # Map children and gender to binary indicators
    df['Children'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    df['GenderMale'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

    # Drop rows where mapping failed
    df = df.dropna(subset=['Children', 'GenderMale'])
    df['Children'] = df['Children'].astype(int)
    df['GenderMale'] = df['GenderMale'].astype(int)

    # Ensure numeric controls are numeric; coerce non-numeric to NaN and drop such rows
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with missing numeric controls or missing outcome
    df = df.dropna(subset=numeric_cols + ['affairs'])

    # Prepare the outcome as integer counts (keep original 'affairs' name)
    # If any non-integer values exist, coerce to int (dataset already codes discrete counts)
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce').fillna(0).astype(int)

    # Mean-center continuous controls for interpretability
    df['age_c'] = df['age'] - df['age'].mean()
    df['yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()
    df['religiousness_c'] = df['religiousness'] - df['religiousness'].mean()
    df['education_c'] = df['education'] - df['education'].mean()
    df['occupation_c'] = df['occupation'] - df['occupation'].mean()
    df['rating_c'] = df['rating'] - df['rating'].mean()

    # Interaction for moderation test: does the effect of Children differ by gender?
    df['Children_x_Male'] = df['Children'] * df['GenderMale']

    # Return only the columns necessary for modeling (plus originals for reference)
    keep_cols = ['affairs', 'Children', 'GenderMale', 'Children_x_Male', 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c']
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a zero-inflated negative binomial regression predicting count of affairs.

    Count model (conditional mean) predictors:
      - Children (main IV)
      - GenderMale
      - Children_x_Male (interaction)
      - age_c, yearsmarried_c, religiousness_c, education_c, occupation_c, rating_c

    Inflation (logit) model predictors (predicting excess zeros):
      - intercept, Children, GenderMale, age_c, yearsmarried_c

    Returns:
      - results object from statsmodels (ZeroInflatedNegativeBinomialPResults)
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    import numpy as np

    df = df.copy()

    # Verify required columns are present
    model_cols = ['Children', 'GenderMale', 'Children_x_Male', 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c']
    missing = [c for c in model_cols + ['affairs'] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Prepare exogenous (count) and exogenous (inflation) matrices
    exog_vars = ['Children', 'GenderMale', 'Children_x_Male', 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c']
    exog = sm.add_constant(df[exog_vars], has_constant='add')

    # Simpler inflation model: intercept + key covariates that predict structural zeros
    exog_infl_vars = ['Children', 'GenderMale', 'age_c', 'yearsmarried_c']
    exog_infl = sm.add_constant(df[exog_infl_vars], has_constant='add')

    endog = df['affairs']

    # Fit Zero-Inflated Negative Binomial (logit inflation)
    zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')

    # Fit the model. Use Newton and reasonable maxiter; fitting can occasionally require increasing maxiter.
    try:
        results = zinb.fit(method='newton', maxiter=100, disp=False)
    except Exception:
        # fallback to default fit if newton fails
        results = zinb.fit(disp=False)

    # Print and return results for inspection
    print(results.summary())
    return results


