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
    """
    Prepare the fishing dataset for modeling.
    - Coerce key columns to numeric
    - Drop rows with missing or invalid exposure/response
    - Create fish_per_hour for descriptive checks
    - Create mean-centered predictors used in the model
    - Create log_hours to use as offset in the GLM

    The final dataframe contains the columns used in the model:
      ['fish_caught', 'hours', 'log_hours', 'livebait', 'child', 'centered_group_size', 'centered_camper', 'fish_per_hour']
    """
    df = df.copy()

    # Ensure numeric types for relevant columns
    for col in ['persons', 'child', 'livebait', 'hours', 'camper', 'fish_caught']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing response or exposure
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove non-positive exposures (hours must be > 0 to define a rate)
    df = df[df['hours'] > 0]

    # Defensive: fill binary columns with 0/1 if close to binary but floats
    # If values are not strictly 0/1 this will coerce to ints where possible
    df['livebait'] = df['livebait'].round().astype(int)
    df['child'] = df['child'].round().astype(int)

    # Create the per-hour rate for descriptive analysis
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Center continuous group-level predictors to improve interpretability
    # Use 'persons' as group size; if persons contains zero or NaN those rows were dropped above
    df['centered_group_size'] = df['persons'] - df['persons'].mean()
    df['centered_camper'] = df['camper'] - df['camper'].mean()

    # Log of hours to use as offset in count models
    df['log_hours'] = np.log(df['hours'])

    # Final: keep only columns needed for modeling (but retain others if desired)
    # We'll return full df with added columns so user can inspect other variables if needed
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model for number of fish caught with hours as exposure (offset).
    Primary model: Negative Binomial GLM with log link and offset = log_hours.
    Predictors: livebait, child, centered_group_size, centered_camper.

    Returns the fitted model results object. If Negative Binomial fails, falls back to Poisson.
    """
    df = df.copy()

    # Ensure the transformed columns exist
    required = ['fish_caught', 'log_hours', 'livebait', 'child', 'centered_group_size', 'centered_camper']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Design matrix
    X = df[['livebait', 'child', 'centered_group_size', 'centered_camper']]
    X = sm.add_constant(X, has_constant='add')
    y = df['fish_caught']
    offset = df['log_hours']

    # Try Negative Binomial (allows for overdispersion relative to Poisson)
    try:
        model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        results_nb = model_nb.fit()
        return results_nb
    except Exception as e:
        # Fall back to Poisson with robust (sandwich) standard errors
        model_p = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        results_p = model_p.fit()
        # attach warning to results object for user's attention
        results_p.model_fallback_warning = (
            'NegativeBinomial failed with error: ' + str(e) + '. Fitted Poisson instead.'
        )
        return results_p


