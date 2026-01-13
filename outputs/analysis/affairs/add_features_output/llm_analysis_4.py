from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (affairs) dataset into a clean dataframe ready for modeling.

    Produces the following new/cleaned columns used by the model:
      - has_children: binary 1/0 mapped from 'children' ('yes' -> 1, 'no' -> 0)
      - gender_male: binary 1/0 mapped from 'gender' ('male' -> 1, 'female' -> 0)
      - affairs: integer count (kept from original, coerced to int)

    Drops rows with missing values in any of the variables used in the model.
    """
    df = df.copy()

    # Map children to binary indicator
    if 'children' in df.columns:
        df['has_children'] = df['children'].map({'yes': 1, 'no': 0})
    else:
        # If column missing, create NA column for downstream drop
        df['has_children'] = np.nan

    # Map gender to binary male indicator
    if 'gender' in df.columns:
        df['gender_male'] = df['gender'].map({'male': 1, 'female': 0})
    else:
        df['gender_male'] = np.nan

    # Ensure key numeric columns are numeric (coerce errors to NaN so they will be dropped)
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Drop rows with missing values in outcome, IV, moderator, or controls used in the model
    required = ['affairs', 'has_children', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required)

    # Cast affairs to integer count (the data uses special codes like 7 and 12; keep as-is)
    df['affairs'] = df['affairs'].astype(int)

    # Optional: verify binary columns are 0/1
    df['has_children'] = df['has_children'].astype(int)
    df['gender_male'] = df['gender_male'].astype(int)

    # Return only columns that will be used downstream (keeps df compact)
    keep_cols = ['affairs', 'has_children', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> object:
    """
    Fit a zero-inflated negative binomial model predicting the count of extramarital affairs.

    Model specification:
      - Outcome (endog): 'affairs'
      - Main predictor: 'has_children'
      - Moderator: 'gender_male' (interaction included to allow the effect of children to differ by gender)
      - Controls: age, yearsmarried, religiousness, education, occupation, rating
      - Zero-inflation (logit) part: includes has_children, gender_male, age, yearsmarried, religiousness to model excess zeros

    Returns the fitted model result object (statsmodels results instance).
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Work on a copy to avoid side effects
    data = df.copy()

    # Build exogenous matrix for the count model (include interaction)
    exog = data[['has_children', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']].copy()
    exog['children_x_male'] = exog['has_children'] * exog['gender_male']
    exog = sm.add_constant(exog, has_constant='add')

    # Exogenous matrix for the inflation (logit) model: use a smaller set to aid convergence
    exog_infl = data[['has_children', 'gender_male', 'age', 'yearsmarried', 'religiousness']].copy()
    exog_infl = sm.add_constant(exog_infl, has_constant='add')

    # Endogenous (dependent) variable
    endog = data['affairs']

    # Fit Zero-Inflated Negative Binomial
    # Use a reliable optimizer and reasonable iteration limits; suppress per-iteration output
    zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')
    try:
        results = zinb.fit(method='bfgs', maxiter=200, disp=False)
    except Exception:
        # fallback: try default fit
        results = zinb.fit(disp=False)

    # Return the fitted results object; the caller can inspect results.summary(), params, conf_int(), etc.
    return results


