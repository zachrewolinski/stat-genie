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
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert to numeric where appropriate, coerce errors to NaN
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing critical values
    df = df.dropna(subset=required)

    # Remove rows with non-positive hours (cannot be used as exposure)
    df = df[df['hours'] > 0]

    # Create observed rate (fish per hour) for descriptive summaries
    df['rate_per_hour'] = df['fish_caught'] / df['hours']

    # Create log-offset of hours for rate modeling
    df['offset_log_hours'] = np.log(df['hours'])

    # Ensure binary covariates are integers 0/1
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Derive group size as a helpful descriptive control (not required by model but useful)
    df['group_size'] = df['persons'] + df['child']

    # Final dataframe contains all columns used later in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits a count regression for fish_caught using hours as exposure (offset).
    Procedure:
    1. Fit Poisson GLM with offset log(hours).
    2. Compute dispersion (Pearson chi-square / df). If overdispersed (>1.5), fit a Negative Binomial GLM.

    Returns a dictionary with fitted models and diagnostics.
    """
    results = {}

    # Required columns check
    required = ['fish_caught', 'offset_log_hours', 'livebait', 'camper', 'persons', 'child']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Design matrix: independent variables and controls
    X = df[['livebait', 'camper', 'persons', 'child']]
    X = sm.add_constant(X, has_constant='add')

    # Offset (log of exposure hours)
    offset = df['offset_log_hours']

    # Fit Poisson GLM with offset
    poisson_model = sm.GLM(df['fish_caught'], X, family=sm.families.Poisson(), offset=offset)
    poisson_res = poisson_model.fit()
    results['poisson'] = poisson_res

    # Compute dispersion: Pearson chi2 / df_resid
    pearson_chi2 = np.sum(poisson_res.resid_pearson**2)
    dispersion = pearson_chi2 / poisson_res.df_resid if poisson_res.df_resid > 0 else np.nan
    results['dispersion'] = float(dispersion)

    # If there is substantial overdispersion, fit Negative Binomial
    # Threshold of 1.5 is a heuristic; users can inspect dispersion and choose
    if dispersion > 1.5:
        nb_model = sm.GLM(df['fish_caught'], X, family=sm.families.NegativeBinomial(), offset=offset)
        nb_res = nb_model.fit()
        results['negative_binomial'] = nb_res
    else:
        results['negative_binomial'] = None

    # Add simple descriptive summaries
    results['n_obs'] = int(df.shape[0])
    results['mean_rate_per_hour'] = float(df['rate_per_hour'].mean())

    return results


