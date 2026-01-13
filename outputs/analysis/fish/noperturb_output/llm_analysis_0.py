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
    Prepare the fishing dataset for modeling.
    Produces the following columns required by the model:
      - fish_caught (int) : dependent count
      - livebait (0/1)
      - camper (0/1)
      - persons (int)
      - child (int)
      - hours (float) : exposure (must be > 0)
      - log_hours (float) : log(hours) used as offset in GLM
      - fish_per_hour (float) : descriptive rate
      - group_size (int) : persons + child

    Drops rows with missing essential values and rows with non-positive hours.
    """
    df = df.copy()

    # Keep only columns we need (if some are missing this will raise later)
    required = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']
    # Ensure columns exist; if not, KeyError will surface to the user
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Convert to numeric types where appropriate
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing essential values
    df = df.dropna(subset=required)

    # Remove rows with non-positive or extremely small hours (cannot take log(0))
    df = df[df['hours'] > 0]

    # Create descriptive rate (fish per hour)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Create group size control variable
    df['group_size'] = df['persons'] + df['child']

    # Ensure binary predictors are integers (0/1)
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Offset for GLM: log(hours)
    df['log_hours'] = np.log(df['hours'])

    # Reset index for a clean dataframe
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count model for fish_caught using exposure (hours) as an offset to estimate rate (fish per hour).

    Strategy:
    1. Fit a Poisson GLM with offset=log_hours.
    2. Compute dispersion = deviance / df_resid. If dispersion > 1.5 (substantial overdispersion), fit a Negative Binomial GLM instead.

    Predictors included: livebait, camper, persons, child (all main effects).

    Returns a dictionary with keys:
      - 'poisson_model': fitted Poisson result object
      - 'dispersion': dispersion statistic (float)
      - 'chosen_model': the model chosen based on dispersion (Poisson or Negative Binomial result)
      - 'nb_model' (only present if Negative Binomial was fit): fitted NB result object

    Note: the returned objects are statsmodels results instances.
    """
    df = df.copy()

    # Basic checks
    if df.shape[0] == 0:
        raise ValueError("Transformed dataframe is empty. Cannot fit model.")
    for col in ['fish_caught', 'log_hours', 'livebait', 'camper', 'persons', 'child']:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe")

    # Design matrix (predictors)
    X = df[['livebait', 'camper', 'persons', 'child']]
    X = sm.add_constant(X, has_constant='add')
    y = df['fish_caught']
    offset = df['log_hours']

    # Fit Poisson GLM first
    poisson = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()

    # Dispersion check: deviance / df_resid
    dispersion = np.nan
    if poisson.df_resid > 0:
        dispersion = float(poisson.deviance / poisson.df_resid)

    results = {
        'poisson_model': poisson,
        'dispersion': dispersion
    }

    # If overdispersed, fit Negative Binomial GLM
    if not np.isnan(dispersion) and dispersion > 1.5:
        nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()
        results['nb_model'] = nb
        results['chosen_model'] = nb
    else:
        results['chosen_model'] = poisson

    return results


