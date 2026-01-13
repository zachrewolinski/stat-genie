from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/add_features_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe into the analysis-ready dataframe.

    Produces the following additional/clean columns used in modeling:
      - group_size: persons + child (both numeric, missing -> 0 before sum)
      - fish_per_hour: descriptive rate = fish_caught / hours
      - log_hours: natural log of hours (used as offset in GLM)

    Drops rows with missing or invalid critical values (fish_caught, hours) and
    ensures numeric types for covariates used in the model.
    """
    df = df.copy()

    # Ensure critical columns exist
    required = ['fish_caught', 'hours']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe")

    # Convert columns that should be numeric to numeric (coerce errors to NaN)
    numeric_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours', 'religiousness', 'age']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing fish_caught or hours
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove non-positive hours (cannot model exposure <= 0)
    df = df[df['hours'] > 0]

    # Create group_size as persons + child (treat missing person/child as 0 for the sum)
    # If persons/child entirely missing in dataset, this will produce NaNs and be handled below
    if 'persons' in df.columns and 'child' in df.columns:
        df['group_size'] = df['persons'].fillna(0) + df['child'].fillna(0)
    elif 'persons' in df.columns:
        df['group_size'] = df['persons'].fillna(0)
    elif 'child' in df.columns:
        df['group_size'] = df['child'].fillna(0)
    else:
        # If neither persons nor child present, create group_size = 1 as fallback (but note in modeling we will drop if missing)
        df['group_size'] = 1

    # Descriptive rate variable
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Offset for GLM: log of hours
    # hours > 0 guaranteed above, so log defined
    df['log_hours'] = np.log(df['hours'])

    # Ensure binary variables are integers (0/1) when present
    for b in ['livebait', 'camper']:
        if b in df.columns:
            # Coerce any non-missing values to 0/1 integers where possible
            df[b] = df[b].fillna(0).astype(int)

    # Keep only rows that have the covariates we plan to use in the main model
    model_cols_required = ['fish_caught', 'log_hours', 'livebait', 'camper', 'group_size', 'religiousness', 'age']
    # If some optional covariates are not present in the dataset, drop them from required list
    model_cols_required = [c for c in model_cols_required if c in df.columns]
    df = df.dropna(subset=model_cols_required)

    # Final copy and return
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression model for fish_caught using exposure (hours) as an offset.

    Primary modeling approach: Negative Binomial GLM with log-link and offset = log_hours.
    This models the rate (fish per hour) while adjusting for covariates.

    Returns the fitted statsmodels results object.
    """
    # Ensure transform has been run and required columns exist
    required = ['fish_caught', 'log_hours']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe. Run transform() first.")

    # Define modeling features (use only those present in df)
    candidate_features = ['livebait', 'camper', 'group_size', 'religiousness', 'age']
    features = [f for f in candidate_features if f in df.columns]

    if len(features) == 0:
        raise ValueError('No model features available in dataframe')

    df_model = df.dropna(subset=features + ['fish_caught', 'log_hours']).copy()

    # Design matrices
    X = sm.add_constant(df_model[features], has_constant='add')
    y = df_model['fish_caught']
    offset = df_model['log_hours']

    # Fit Negative Binomial GLM (models counts with exposure offset)
    try:
        model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        results = model_nb.fit()
    except Exception as e:
        # Fallback to Poisson with robust SEs if NB fails
        model_p = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        results = model_p.fit(cov_type='HC0')

    # Print and return results
    print(results.summary())
    return results


