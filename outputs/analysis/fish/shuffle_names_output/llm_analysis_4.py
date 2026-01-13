from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/shuffle_names_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure numeric types for core columns; coerce errors to NaN
    for col in ['fish_caught', 'hours', 'persons', 'child', 'livebait', 'camper']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core variables needed for modeling
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove/flag rows with non-positive hours to avoid division by zero in rate calculation
    df = df[df['hours'] > 0]

    # Derive fish per hour (rate)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Derive a binary indicator for presence of any camper in the group
    # If camper column encodes number of campers, treat >0 as having a camper
    if 'camper' in df.columns:
        df['has_camper'] = (df['camper'] > 0).astype(int)
    else:
        # if camper not present, create a missing indicator column filled with 0
        df['has_camper'] = 0

    # Ensure livebait and child are binary 0/1 (coerce nonzero to 1)
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].fillna(0).astype(int)
        df['livebait'] = df['livebait'].apply(lambda x: 1 if x != 0 else 0)
    else:
        df['livebait'] = 0

    if 'child' in df.columns:
        df['child'] = df['child'].fillna(0).astype(int)
        df['child'] = df['child'].apply(lambda x: 1 if x != 0 else 0)
    else:
        df['child'] = 0

    # Keep only the columns required for modeling (plus original fish_caught and camper for transparency)
    keep_cols = ['fish_caught', 'hours', 'fish_per_hour', 'livebait', 'has_camper', 'persons', 'child', 'camper']
    # Some of these columns might not exist; retain the ones that do
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Expecting df to be the output of transform(df)
    # Import local references (statsmodels already available globally)
    import numpy as np
    import statsmodels.api as sm

    # Prepare predictors for both models
    # Use the same set of covariates: livebait, has_camper, persons, child
    predictors = []
    for col in ['livebait', 'has_camper', 'persons', 'child']:
        if col in df.columns:
            predictors.append(col)
    if len(predictors) == 0:
        raise ValueError('No predictor columns available for modeling.')

    X = df[predictors].copy()
    X = sm.add_constant(X, has_constant='add')

    results = {}

    # 1) Linear regression on fish_per_hour (OLS). Use robust (HC3) standard errors to protect against heteroskedasticity.
    if 'fish_per_hour' in df.columns:
        ols_model = sm.OLS(df['fish_per_hour'], X).fit(cov_type='HC3')
        results['ols_fish_per_hour'] = ols_model
    else:
        results['ols_fish_per_hour'] = None

    # 2) Poisson GLM for counts with log(hours) as offset (models rate while using fish_caught as response)
    #    Poisson with offset is a canonical way to model counts per exposure; robust covariance is used.
    if 'fish_caught' in df.columns and 'hours' in df.columns:
        # Avoid negative or zero fish_caught values for Poisson link: Poisson can accept non-integer positive values
        # (interpreting them as rates scaled by exposure). Statsmodels accepts them.
        offset = np.log(df['hours'].values)
        glm_model = sm.GLM(df['fish_caught'], X, family=sm.families.Poisson(), offset=offset).fit(cov_type='HC3')
        results['poisson_glm_rate'] = glm_model
    else:
        results['poisson_glm_rate'] = None

    # Return results dict containing fitted model objects. Callers can print summaries:
    # For example: print(results['ols_fish_per_hour'].summary())
    return results


