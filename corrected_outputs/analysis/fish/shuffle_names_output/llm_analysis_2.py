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
    # Work on a copy to avoid mutating original
    df = df.copy()

    # Ensure numeric types where appropriate
    for col in ['fish_caught', 'hours', 'persons', 'camper', 'livebait', 'child']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing outcome or exposure (hours) or necessary predictors
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Keep only rows with positive hours (cannot compute per-hour rate otherwise)
    df = df[df['hours'] > 0]

    # Create the primary dependent variable: fish caught per hour
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Add a small constant and log-transform to reduce skew for regression
    # small constant chosen to allow log when fish_per_hour is zero
    eps = 1e-6
    df['log_fish_per_hour'] = np.log(df['fish_per_hour'] + eps)

    # Ensure binary indicators are integers (0/1)
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].fillna(0).astype(int)
    else:
        # If missing, create a default column of zeros
        df['livebait'] = 0

    if 'child' in df.columns:
        df['child'] = df['child'].fillna(0).astype(int)
    else:
        df['child'] = 0

    # Standardize continuous predictors (persons and camper) for interpretability
    # Use population standard deviation (ddof=0) for stability in small samples
    if 'persons' in df.columns:
        persons_mean = df['persons'].mean()
        persons_std = df['persons'].std(ddof=0)
        # Avoid division by zero
        if persons_std == 0 or np.isnan(persons_std):
            df['persons_s'] = 0.0
        else:
            df['persons_s'] = (df['persons'] - persons_mean) / persons_std
    else:
        # If column missing, create zeros
        df['persons_s'] = 0.0

    if 'camper' in df.columns:
        camper_mean = df['camper'].mean()
        camper_std = df['camper'].std(ddof=0)
        if camper_std == 0 or np.isnan(camper_std):
            df['camper_s'] = 0.0
        else:
            df['camper_s'] = (df['camper'] - camper_mean) / camper_std
    else:
        df['camper_s'] = 0.0

    # Interaction term between livebait and standardized group size
    df['livebait_persons'] = df['livebait'] * df['persons_s']

    # Keep only columns needed for modeling plus original identifiers if present
    model_cols = ['fish_per_hour', 'log_fish_per_hour', 'livebait', 'persons_s', 'camper_s', 'child', 'livebait_persons']
    # add any additional columns present in df that might be useful later (not required)
    available = [c for c in model_cols if c in df.columns]
    return df[available].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # df is the transformed dataframe returned by transform()
    # We fit an OLS model predicting the log of fish-per-hour using the predictors.
    # Using log(fish_per_hour) models multiplicative effects on the rate and reduces skew.

    # Ensure required columns present
    required = ['log_fish_per_hour', 'livebait', 'persons_s', 'camper_s', 'child', 'livebait_persons']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare design matrix
    X = df[['livebait', 'persons_s', 'camper_s', 'child', 'livebait_persons']].copy()
    X = sm.add_constant(X)
    y = df['log_fish_per_hour']

    # Fit OLS on the log rate
    model_res = sm.OLS(y, X).fit()

    # Return the fitted model object (statsmodels RegressionResults)
    return model_res


