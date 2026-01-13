from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/replace_with_rvs_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations performed:
      - Drop rows with missing essential values (fish_caught, hours, livebait, camper, persons, child).
      - Remove rows with nonpositive hours (can't compute log exposure); if any hours are extremely small but >0 they are kept.
      - Create log_hours (offset) = log(hours).
      - Create fish_per_hour = fish_caught / hours for descriptive checks.
      - Create total_people = persons + child.
      - Ensure binary columns are integer-typed (0/1).
    Returns the dataframe containing all columns required for modeling.
    """

    # Copy to avoid modifying original
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing required values
    df = df.dropna(subset=required_cols)

    # Convert numeric columns to appropriate types
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce').astype('Int64')
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce').astype('Int64')
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df['child'] = pd.to_numeric(df['child'], errors='coerce')

    # Drop rows that became NaN after coercion
    df = df.dropna(subset=['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child'])

    # Remove nonpositive hours (can't take log); if such rows exist, drop them as exposure is invalid
    df = df[df['hours'] > 0]

    # Create log_hours for offset and fish_per_hour for descriptive purposes
    df['log_hours'] = np.log(df['hours'])
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Create total_people for potential alternative specifications/diagnostics
    df['total_people'] = df['persons'] + df['child']

    # Ensure binary columns are 0/1 integers (in case they were booleans or other numerics)
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Optional: filter out extreme outliers for descriptive robustness checks (not applied by default)
    # e.g., keep rows with fish_caught within a reasonable bound if desired. Here we keep all data.

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression model for fish counts using exposure (hours) as an offset to estimate fish per hour.

    Primary model: Negative Binomial GLM to account for overdispersion often present in count data.
    Model specification: fish_caught ~ livebait + camper + persons + child
    Offset: log_hours (so the model estimates rate per hour)

    Returns:
      - results: the fitted GLMResults object from statsmodels
      - diagnostics: a small dict with observed mean/variance of counts and a short overdispersion check
    """

    # Ensure the transformed dataframe contains the needed columns
    needed = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'log_hours', 'hours']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare design matrix (exogenous variables) and endogenous variable
    exog = df[['livebait', 'camper', 'persons', 'child']].astype(float)
    exog = sm.add_constant(exog, has_constant='add')
    endog = df['fish_caught'].astype(float)
    offset = df['log_hours'].astype(float)

    # Fit Negative Binomial GLM with offset to model counts with exposure
    # If NegativeBinomial fails to converge for some datasets, consider Poisson as fallback.
    try:
        model_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(), offset=offset)
        results = model_nb.fit()
    except Exception as e:
        # Fallback to Poisson if NB fails
        model_p = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset)
        results = model_p.fit()
        results.model_fallback_warning = str(e)

    # Simple diagnostics: mean and variance of counts and empirical overdispersion ratio
    mean_count = endog.mean()
    var_count = endog.var()
    overdispersion_ratio = var_count / mean_count if mean_count > 0 else np.nan

    diagnostics = {
        'mean_count': float(mean_count),
        'var_count': float(var_count),
        'overdispersion_ratio': float(overdispersion_ratio),
        'n_obs': int(len(df))
    }

    # Also compute predicted rate (predicted expected count divided by hours)
    pred_counts = results.predict(exog=exog, offset=offset)
    df = df.copy()
    df['predicted_count'] = pred_counts
    df['predicted_rate_per_hour'] = df['predicted_count'] / df['hours']

    # Attach diagnostics and the transformed df to results for convenience
    results.diagnostics = diagnostics
    results.predictions = df[['predicted_count', 'predicted_rate_per_hour']]

    return results


