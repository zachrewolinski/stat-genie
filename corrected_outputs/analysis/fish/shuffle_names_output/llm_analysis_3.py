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

    # Ensure numeric types; coerce errors to NaN
    for col in ['persons', 'child', 'livebait', 'hours', 'camper', 'fish_caught']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing essential values
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove rows with non-positive hours (can't form a rate)
    df = df[df['hours'] > 0]

    # Force binary columns to 0/1 (if they are floats like 0.0/1.0)
    # If values are not exactly 0/1 they will be coerced to int (may raise ValueError if floats are NaN but NaNs removed above)
    df['livebait'] = df['livebait'].fillna(0).astype(int)
    df['child'] = df['child'].fillna(0).astype(int)

    # Derived variables
    # Observed rate (for description / plotting)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Offset for GLM: log of exposure hours
    df['log_hours'] = np.log(df['hours'])

    # Standardize continuous predictors (persons and camper) for numeric stability and easier coefficient interpretation
    # Use population std (ddof=0) but guard against zero std
    if 'persons' in df.columns:
        persons_std = df['persons'].std(ddof=0)
        if np.isclose(persons_std, 0) or np.isnan(persons_std):
            df['persons_z'] = df['persons'] - df['persons'].mean()
        else:
            df['persons_z'] = (df['persons'] - df['persons'].mean()) / persons_std
    else:
        df['persons_z'] = 0.0

    if 'camper' in df.columns:
        camper_std = df['camper'].std(ddof=0)
        if np.isclose(camper_std, 0) or np.isnan(camper_std):
            df['camper_z'] = df['camper'] - df['camper'].mean()
        else:
            df['camper_z'] = (df['camper'] - df['camper'].mean()) / camper_std
    else:
        df['camper_z'] = 0.0

    # Return transformed dataframe with all columns required by the model
    # Expected final columns used in modeling: ['fish_caught','hours','log_hours','fish_per_hour','livebait','persons_z','camper_z','child']
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a regression for fish caught with an exposure offset so coefficients represent multiplicative effects on the rate (fish per hour).

    Primary approach: Negative-binomial GLM with log link and offset=log(hours) to account for exposure.
    If NegativeBinomial fails to converge, fall back to Poisson GLM.

    Returns:
        A fitted statsmodels results object (GLMResults). Caller can inspect summary(), params, predict(), etc.
    """
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['fish_caught', 'log_hours', 'livebait', 'persons_z', 'camper_z', 'child']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Define predictors (main effects)
    predictors = ['livebait', 'persons_z', 'camper_z', 'child']

    # Build design matrix with constant
    X = sm.add_constant(df[predictors], has_constant='add')
    y = df['fish_caught']
    offset = df['log_hours']

    # Try Negative Binomial GLM first (handles overdispersion)
    try:
        model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        results = model_nb.fit()
    except Exception:
        # Fallback: Poisson with offset
        model_p = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        results = model_p.fit()

    # Return fitted results object (caller can call results.summary())
    return results


