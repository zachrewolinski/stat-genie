from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/anonymize_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to meaningful names
    df = df.rename(columns={
        'feature1': 'FishCaught',
        'feature2': 'LiveBait',
        'feature3': 'HasCamper',
        'feature4': 'NumAdults',
        'feature5': 'NumChildren',
        'feature6': 'Hours'
    }).copy()

    # Keep only rows with non-missing values for key variables
    df = df.dropna(subset=['FishCaught', 'LiveBait', 'HasCamper', 'NumAdults', 'NumChildren', 'Hours'])

    # Ensure numeric types
    df['FishCaught'] = pd.to_numeric(df['FishCaught'], errors='coerce')
    df['LiveBait'] = pd.to_numeric(df['LiveBait'], errors='coerce').astype(int)
    df['HasCamper'] = pd.to_numeric(df['HasCamper'], errors='coerce').astype(int)
    df['NumAdults'] = pd.to_numeric(df['NumAdults'], errors='coerce')
    df['NumChildren'] = pd.to_numeric(df['NumChildren'], errors='coerce')
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce')

    # Drop any rows made NA by coercion
    df = df.dropna(subset=['FishCaught', 'LiveBait', 'HasCamper', 'NumAdults', 'NumChildren', 'Hours'])

    # Filter out non-positive hours (cannot use as exposure)
    df = df[df['Hours'] > 0].copy()

    # Derived columns
    # FishPerHour: descriptive rate (float)
    df['FishPerHour'] = df['FishCaught'] / df['Hours']

    # Log of hours used as offset in count models
    df['logHours'] = np.log(df['Hours'])

    # Optionally create group size for diagnostics (not used directly in main model)
    df['GroupSize'] = df['NumAdults'] + df['NumChildren']

    # Final columns required for modeling: FishCaught, LiveBait, HasCamper, NumAdults, NumChildren, Hours, logHours, FishPerHour
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build design matrix (predictors). Use additive model for rate of fish caught per hour.
    X = df[['LiveBait', 'HasCamper', 'NumAdults', 'NumChildren']].copy()
    X = sm.add_constant(X, has_constant='add')

    # Response: count of fish caught. Use Negative Binomial GLM with log(Hours) as offset to model rate per hour.
    # If NegativeBinomial family is unavailable in the environment, fallback to Poisson and warn user.
    try:
        family = sm.families.NegativeBinomial()
    except Exception:
        family = sm.families.Poisson()

    model_glm = sm.GLM(endog=df['FishCaught'], exog=X, family=family, offset=df['logHours'])
    results = model_glm.fit()

    # Return the fitted results object (has .summary(), .params, .bse, etc.)
    return results


