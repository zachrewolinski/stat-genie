from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/fish/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw fishing-visit data to a modeling-ready dataframe.
    Produces the following additional columns required by the model:
      - group_size: persons + child
      - fish_per_hour: fish_caught / hours (descriptive)
      - log_hours: natural log of hours (offset / exposure)

    Drops rows with missing essential fields or non-positive hours.
    Returns: transformed dataframe containing at minimum the columns:
      ['fish_caught','livebait','camper','persons','child','hours','group_size','fish_per_hour','log_hours']
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing values in essential columns
    df = df.dropna(subset=required)

    # Remove rows with non-positive or extremely small hours (can't take log or not meaningful)
    # Keep only rows with hours > 0
    df = df[df['hours'] > 0]

    # Create group size (total fishers = adults + children)
    df['group_size'] = df['persons'] + df['child']

    # Descriptive rate: fish per hour
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Offset (log exposure) for count models estimating rate per hour
    df['log_hours'] = np.log(df['hours'])

    # Ensure binary variables are integers 0/1
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # If group_size is zero for any row (shouldn't be), drop those rows to avoid modeling issues
    df = df[df['group_size'] > 0]

    # Return dataframe with all original plus derived columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count regression for fish_caught with park-hours as exposure to estimate rate per hour.

    Workflow:
      1. Fit a Poisson GLM with offset = log_hours and predictors: livebait, camper, group_size.
      2. Compute Pearson dispersion statistic. If dispersion indicates overdispersion (> 1.5), fit a Negative Binomial GLM.
      3. Return a dictionary containing the fitted models and diagnostics. The chosen model under overdispersion will be the Negative Binomial; otherwise Poisson.

    Returns a dict with keys:
      - 'poisson_model': fitted Poisson results (statsmodels object)
      - 'dispersion': Pearson dispersion estimate
      - 'final_model': chosen fitted model (Poisson or NegativeBinomial)
      - 'final_model_name': string naming chosen model
      - 'aic': AIC of the chosen model
      - 'summary_str': textual model summary for quick inspection
    """
    import statsmodels.api as sm

    # Required columns check
    for col in ['fish_caught', 'livebait', 'camper', 'group_size', 'log_hours']:
        if col not in df.columns:
            raise ValueError(f"Required column for modeling not found: {col}")

    # Prepare outcome, design matrix, and offset (exposure)
    y = df['fish_caught'].astype(float)
    X = df[['livebait', 'camper', 'group_size']].astype(float)
    X = sm.add_constant(X, has_constant='add')
    offset = df['log_hours'].astype(float)

    # Fit Poisson GLM with offset
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()

    # Pearson dispersion: sum(pearson_resid^2) / df_resid
    mu = poisson_model.mu
    # Protect against zero mu values (shouldn't happen for Poisson fit) by clipping
    mu_clip = np.clip(mu, 1e-8, None)
    pearson_resid = (y - mu_clip) / np.sqrt(mu_clip)
    pearson_chi2 = np.sum(pearson_resid ** 2)
    dispersion = pearson_chi2 / poisson_model.df_resid if poisson_model.df_resid > 0 else np.nan

    # Decide whether to use Negative Binomial based on dispersion
    chosen_model = poisson_model
    chosen_name = 'poisson'

    if np.isfinite(dispersion) and dispersion > 1.5:
        # Overdispersion detected: try Negative Binomial
        try:
            nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()
            # Use AIC to choose between Poisson and NB if both succeeded
            if nb_model.aic < poisson_model.aic:
                chosen_model = nb_model
                chosen_name = 'negbin'
            else:
                chosen_model = poisson_model
                chosen_name = 'poisson'
        except Exception as e:
            # If NB fails for any reason, keep Poisson and warn (returned in summary)
            chosen_model = poisson_model
            chosen_name = 'poisson_nb_failed'
    else:
        chosen_model = poisson_model
        chosen_name = 'poisson'

    results = {
        'poisson_model': poisson_model,
        'dispersion': dispersion,
        'final_model': chosen_model,
        'final_model_name': chosen_name,
        'aic': chosen_model.aic,
        'summary_str': chosen_model.summary().as_text()
    }

    return results


