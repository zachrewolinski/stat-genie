from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/noperturb_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fishing dataset to the modeling dataframe.

    Steps:
    - Ensure numeric types for key columns and coerce invalid entries to NaN
    - Drop rows missing fish_caught or hours and keep only positive hours
    - Create fish_per_hour and log_hours (for offset)
    - Create group_size and center numeric predictors (persons, child, group_size)
    - Ensure livebait and camper are integer binaries

    The returned dataframe contains the exact columns used by the model:
    ['fish_caught', 'hours', 'log_hours', 'fish_per_hour', 'livebait', 'camper',
     'persons', 'child', 'group_size', 'persons_c', 'child_c', 'group_size_c']
    """
    df = df.copy()

    # Ensure numeric types for key columns
    for col in ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If a column is missing, create it as NaNs so downstream code fails clearly
            df[col] = np.nan

    # Drop rows that cannot be used for rate estimation
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove zero or negative hours (cannot take log and not meaningful exposure)
    df = df[df['hours'] > 0].copy()

    # Per-hour outcome and offset
    df['fish_per_hour'] = df['fish_caught'] / df['hours']
    df['log_hours'] = np.log(df['hours'])

    # Ensure binary indicators are ints (0/1)
    # If they contain other codes, coercion above will produce NaN; fill NaN with 0 conservatively
    df['livebait'] = df['livebait'].fillna(0).astype(int)
    df['camper'] = df['camper'].fillna(0).astype(int)

    # Construct group size and center numerical predictors
    df['persons'] = df['persons'].fillna(0)
    df['child'] = df['child'].fillna(0)
    df['group_size'] = df['persons'] + df['child']

    # Center continuous predictors to improve interpretability of intercept
    df['persons_c'] = df['persons'] - df['persons'].mean()
    df['child_c'] = df['child'] - df['child'].mean()
    df['group_size_c'] = df['group_size'] - df['group_size'].mean()

    # Keep only columns needed for modeling (but preserve extras for diagnostics)
    # Return full df with the transformed columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression to estimate fish-catch rate per hour and effects of predictors.

    Approach:
    - Use a Poisson GLM with log(hours) as an offset to model the count outcome as a rate per hour
    - Check for overdispersion using Pearson chi-square / df_resid. If substantial overdispersion
      is detected (dispersion > 1.5), refit with a Negative Binomial family.
    - Predictors: livebait (binary), camper (binary), group_size_c (centered). Controls persons_c and child_c
      are represented within group_size_c; if desired they can be added to exog_cols.

    Returns:
    - The fitted statsmodels result object (either Poisson or Negative Binomial). The result will have
      an attached attribute `.dispersion` containing the Pearson dispersion estimate computed from the
      initial Poisson fit.
    """
    df = df.copy()

    # Define predictors to include in the linear predictor
    exog_cols = ['livebait', 'camper', 'group_size_c']

    # Ensure columns exist
    for c in exog_cols:
        if c not in df.columns:
            raise ValueError(f"Required column for modeling is missing: {c}")

    # Prepare matrices
    X = sm.add_constant(df[exog_cols])
    y = df['fish_caught']
    offset = df['log_hours']

    # Fit Poisson GLM with offset
    poisson_mod = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_res = poisson_mod.fit()

    # Predicted mean from Poisson model
    mu = poisson_res.predict(X, offset=offset)

    # Pearson chi-square dispersion estimate
    # add small epsilon to mu to avoid division by zero in pathological cases
    eps = 1e-8
    pearson_chi2 = (((y - mu) ** 2) / (mu + eps)).sum()
    df_resid = poisson_res.df_resid if hasattr(poisson_res, 'df_resid') else (len(y) - X.shape[1])
    dispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan

    # Attach dispersion to results for diagnostics
    setattr(poisson_res, 'dispersion', dispersion)

    # If overdispersion is present, refit with Negative Binomial
    if (not np.isnan(dispersion)) and (dispersion > 1.5):
        nb_mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        nb_res = nb_mod.fit()
        setattr(nb_res, 'dispersion_from_poisson', dispersion)
        return nb_res
    else:
        return poisson_res


