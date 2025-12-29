from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/fish/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure relevant columns are numeric; coerce errors to NaN
    numeric_cols = ['persons', 'child', 'livebait', 'hours', 'camper', 'fish_caught']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the essential outcome or exposure
    if 'fish_caught' in df.columns and 'hours' in df.columns:
        df = df.dropna(subset=['fish_caught', 'hours'])
    else:
        raise ValueError('Input dataframe must contain fish_caught and hours columns')

    # Remove observations with non-positive hours to allow log(offset)
    df = df[df['hours'] > 0]

    # Derive the primary dependent variable: fish per hour
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Log of hours for use as an offset in rate models
    df['log_hours'] = np.log(df['hours'])

    # Clean binary indicators: coerce NaNs to 0 when reasonable and ensure integer dtype for binary flags
    # (If a user prefers to drop missing binary flags instead, modify accordingly.)
    if 'child' in df.columns:
        df['child'] = df['child'].fillna(0).astype(int)
    if 'livebait' in df.columns:
        # livebait should be binary 0/1; coerce non-zero to 1
        df['livebait'] = df['livebait'].fillna(0).astype(int).clip(lower=0)

    # Ensure camper and persons are numeric and drop rows with missing predictors
    predictor_cols = [c for c in ['persons', 'camper'] if c in df.columns]
    if predictor_cols:
        df[predictor_cols] = df[predictor_cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows where the derived fish_per_hour is not finite
    df = df[np.isfinite(df['fish_per_hour'])]

    # Final: drop rows missing any model-required columns (defensive)
    required_for_model = ['fish_caught', 'hours', 'log_hours', 'fish_per_hour', 'livebait', 'persons', 'child', 'camper']
    present_required = [c for c in required_for_model if c in df.columns]
    df = df.dropna(subset=present_required)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit two complementary models to estimate factors influencing catch rate (fish per hour):
      1) Primary model: GLM with Gamma family and log link, using log(hours) as an offset. This models fish_caught as a Gamma-distributed positive outcome and estimates multiplicative effects on the catch rate (fish/hour).
      2) Secondary model: OLS on fish_per_hour (continuous) with robust SEs as a simple linear approximation.

    The predictor set used: livebait, persons, child, camper.
    Returns a dictionary with fitted model objects: {'glm_results': glm_results, 'ols_results': ols_results}
    """
    df = df.copy()

    # Ensure required columns are present
    required = ['fish_caught', 'hours', 'log_hours', 'fish_per_hour', 'livebait', 'persons', 'child', 'camper']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Dataframe missing required columns for modeling: {missing}')

    # Drop rows with missing values in the predictors/outcome
    model_df = df.dropna(subset=['fish_caught', 'log_hours', 'fish_per_hour', 'livebait', 'persons', 'child', 'camper'])

    # Prepare design matrix
    predictors = ['livebait', 'persons', 'child', 'camper']
    X = sm.add_constant(model_df[predictors])

    # 1) GLM: Gamma family with log link and log(hours) as offset to model rate per hour
    gamma_family = sm.families.Gamma(link=sm.families.links.log())
    glm_model = sm.GLM(model_df['fish_caught'], X, family=gamma_family, offset=model_df['log_hours'])
    glm_results = glm_model.fit()

    # 2) OLS on fish_per_hour as an alternative specification (robust SEs)
    ols_model = sm.OLS(model_df['fish_per_hour'], X)
    ols_results = ols_model.fit(cov_type='HC3')

    # Return both fitted results so users can inspect coefficients, standard errors, CIs, and goodness-of-fit
    return {
        'glm_results': glm_results,
        'ols_results': ols_results
    }


