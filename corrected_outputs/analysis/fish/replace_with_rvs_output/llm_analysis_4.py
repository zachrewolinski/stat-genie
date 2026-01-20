from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/replace_with_rvs_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fishing dataset to a modeling-ready dataframe.

    Creates centered versions of person counts, drops rows with missing or invalid hours,
    enforces binary dtypes for livebait and camper, and computes a per-hour rate column
    for descriptive summaries.

    Required output columns for modeling: fish_caught, livebait, camper, persons_c, child_c, hours, log_hours, fish_per_hour
    """
    # Work on a copy
    df = df.copy()

    # Drop rows with essential missing data
    df = df.dropna(subset=['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child'])

    # Remove rows with non-positive or effectively zero hours (cannot use as exposure)
    df = df[df['hours'] > 0]

    # Ensure binary columns are integers 0/1
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Ensure count columns are integers
    df['fish_caught'] = df['fish_caught'].astype(int)
    df['persons'] = df['persons'].astype(int)
    df['child'] = df['child'].astype(int)

    # Create a log-hours column to use as an offset in the count model
    df['log_hours'] = np.log(df['hours'])

    # Center the person count variables to aid interpretation of intercept
    df['persons_c'] = df['persons'] - df['persons'].mean()
    df['child_c'] = df['child'] - df['child'].mean()

    # Descriptive rate column (fish per hour) for summaries/diagnostics
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Return the transformed dataframe containing all columns needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression for fish_caught with hours as exposure.

    Procedure:
    1. Fit a Poisson GLM with offset = log_hours.
    2. Compute the Pearson dispersion statistic. If there is substantial overdispersion
       (dispersion > 1.5) fit a Negative Binomial GLM instead.
    3. Return the fitted models and diagnostics.

    Model formula: fish_caught ~ livebait + camper + persons_c + child_c
    Offset: log_hours (log of hours) to model rate per hour.
    """
    # Build design matrix
    exog = df[['livebait', 'camper', 'persons_c', 'child_c']]
    exog = sm.add_constant(exog, has_constant='add')
    endog = df['fish_caught']
    offset = df['log_hours']

    # Fit Poisson GLM with offset
    poisson_model = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset).fit()

    # Compute Pearson dispersion: sum(resid_pearson^2) / df_resid
    pearson_chi2 = np.sum(poisson_model.resid_pearson**2)
    df_resid = poisson_model.df_resid if poisson_model.df_resid is not None else max(len(endog) - exog.shape[1], 1)
    dispersion = pearson_chi2 / df_resid

    # If overdispersed, fit Negative Binomial GLM
    if dispersion > 1.5:
        nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(), offset=offset).fit()
        chosen_model = nb_model
        chosen_family = 'NegativeBinomial'
    else:
        chosen_model = poisson_model
        chosen_family = 'Poisson'

    # Prepare a small result dictionary for downstream use
    results = {
        'poisson_model': poisson_model,
        'chosen_model': chosen_model,
        'chosen_family': chosen_family,
        'dispersion': dispersion,
        'n_obs': int(len(df)),
        'formula': 'fish_caught ~ livebait + camper + persons_c + child_c (offset=log_hours)'
    }

    return results


