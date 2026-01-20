from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/noperturb_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fishing-trip dataframe for modeling.
    Produces the following new/clean columns used in the model:
      - total_people: persons + child
      - fish_per_hour: fish_caught / hours (for descriptive checks)
      - log_hours: log(hours) (not used directly as predictor but useful diagnostically)

    Drops rows with missing or invalid essential fields and removes rows with non-positive hours.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where appropriate
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce')
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce')
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df['child'] = pd.to_numeric(df['child'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')

    # Drop any newly introduced NaNs
    df = df.dropna(subset=required_cols)

    # Remove rows with non-positive or extremely small hours (exposure must be > 0)
    df = df[df['hours'] > 0]

    # Cast binary indicators to int (0/1)
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Derive total people (adults + children)
    df['total_people'] = df['persons'] + df['child']

    # Create a per-hour rate column for descriptive work
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Log hours for diagnostics/plots
    # (not included as a predictor because hours is used as exposure in the count model)
    df['log_hours'] = np.log(df['hours'])

    # Optional: remove groups with zero total_people (shouldn't happen, but be safe)
    df = df[df['total_people'] > 0]

    # Reset index for downstream convenience
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a Negative Binomial GLM for fish counts with hours as exposure.

    Model specification (on counts):
      fish_caught ~ livebait + camper + total_people
    with exposure = hours (log-exposure is handled internally by statsmodels).

    A negative binomial is chosen to allow for overdispersion relative to Poisson.

    Returns the fitted results object (statsmodels GLMResults).
    """
    # Make a copy to avoid modifying caller's dataframe
    df = df.copy()

    # Select predictors and add constant
    X = df[['livebait', 'camper', 'total_people']]
    X = sm.add_constant(X, has_constant='add')

    # Endogenous variable
    y = df['fish_caught']

    # Exposure (must be positive)
    exposure = df['hours']

    # Fit Negative Binomial GLM with exposure
    # Note: statsmodels' NegativeBinomial family in GLM context models counts and accepts exposure.
    model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), exposure=exposure)
    results = model_glm.fit()

    # For convenience, attach a small diagnostic table (overdispersion check) to results
    try:
        pearson_chi2 = ((results.resid_pearson ** 2).sum())
        df_resid = results.df_resid
        results.pearson_chi2 = pearson_chi2
        results.pearson_chi2_per_df = pearson_chi2 / max(df_resid, 1)
    except Exception:
        # If resid_pearson not available for some reason, skip
        pass

    return results


