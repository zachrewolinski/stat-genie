from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/shuffle_names_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric types for key columns; coerce errors to NaN
    numeric_cols = ['persons', 'child', 'livebait', 'hours', 'camper', 'fish_caught']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing essential variables (hours or fish_caught) and remove non-positive hours
    df = df.dropna(subset=['hours', 'fish_caught'])
    df = df[df['hours'] > 0]

    # Create cleaned predictor variables with clear names used in modeling
    # GroupSize: number of persons in the group (may be 0 or missing in original; keep numeric)
    df['GroupSize'] = df['persons']

    # ChildPresent: binary indicator for presence of children
    # If original has values other than 0/1, coerce to 1 for any nonzero
    df['ChildPresent'] = df['child'].fillna(0).astype(int).clip(0,1)

    # LiveBait: binary indicator for use of live bait
    df['LiveBait'] = df['livebait'].fillna(0).astype(int).clip(0,1)

    # CamperCount: numeric count of campers in group (original 'camper' column)
    df['CamperCount'] = df['camper'].fillna(0).astype(int)
    # HasCamper: binary indicator if any camper present
    df['HasCamper'] = (df['CamperCount'] > 0).astype(int)

    # Compute the primary outcome: fish caught per hour
    df['FishPerHour'] = df['fish_caught'] / df['hours']

    # Log-transform the rate to reduce skew. Add a small constant to avoid log(0).
    eps = 0.001
    df['LogFishPerHour'] = np.log(df['FishPerHour'] + eps)

    # Winsorize the log-rate at the 1st and 99th percentiles to reduce influence of extreme outliers
    if df['LogFishPerHour'].notna().sum() > 0:
        lower = df['LogFishPerHour'].quantile(0.01)
        upper = df['LogFishPerHour'].quantile(0.99)
        df['LogFishPerHour'] = df['LogFishPerHour'].clip(lower, upper)

    # Keep only columns needed for modeling plus original for traceability
    model_cols = ['persons', 'child', 'livebait', 'hours', 'camper', 'fish_caught',
                  'GroupSize', 'ChildPresent', 'LiveBait', 'CamperCount', 'HasCamper',
                  'FishPerHour', 'LogFishPerHour']
    # Some of these might not exist if original data lacked them; select existing
    existing = [c for c in model_cols if c in df.columns]
    return df[existing]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits an OLS model predicting log fish-per-hour using primary predictors and controls.
    Returns the fitted OLS results object (with robust standard errors).
    """
    # Copy to avoid side effects
    data = df.copy()

    # Select predictors and drop rows with missing values in these columns or in the outcome
    predictors = ['LiveBait', 'ChildPresent', 'GroupSize', 'CamperCount', 'HasCamper']
    required = ['LogFishPerHour'] + predictors
    data = data.dropna(subset=required)

    # Design matrix
    X = data[predictors]
    X = sm.add_constant(X)
    y = data['LogFishPerHour']

    # Fit OLS with robust (HC3) standard errors to account for heteroskedasticity
    model_fit = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted model object (contains params, summary, etc.)
    return model_fit


