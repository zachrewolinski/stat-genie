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
    Transform the raw fishing dataset into the final dataframe used for modeling.

    Produces the following columns required by the model:
      - fish_caught (original count)
      - livebait (0/1)
      - camper (0/1)
      - persons_c (mean-centered adults)
      - child_c (mean-centered children)
      - total_people (persons + child)
      - hours (positive float)
      - log_hours (log of hours, for offset)
      - catch_rate (fish_caught / hours)

    Drops rows with missing essential data and rows with non-positive hours.
    """
    df = df.copy()

    # Drop rows missing essential fields
    essential_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    df = df.dropna(subset=essential_cols)

    # Ensure numeric types
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce')
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce')
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df['child'] = pd.to_numeric(df['child'], errors='coerce')

    # Drop any rows that became NaN after coercion
    df = df.dropna(subset=essential_cols)

    # Remove non-positive hours (cannot take log or use as exposure)
    df = df[df['hours'] > 0]

    # Basic derived columns
    df['total_people'] = df['persons'] + df['child']

    # Mean-center persons and child for interpretability
    df['persons_c'] = df['persons'] - df['persons'].mean()
    df['child_c'] = df['child'] - df['child'].mean()

    # Ensure binary indicators are integers 0/1
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Log of hours for use as offset
    df['log_hours'] = np.log(df['hours'].astype(float))

    # Observed catch rate (fish per hour) as a descriptive variable
    df['catch_rate'] = df['fish_caught'] / df['hours']

    # Final safety: drop any rows with infinite or NaN values created
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['fish_caught', 'hours', 'log_hours', 'livebait', 'camper', 'persons_c', 'child_c'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count regression model of fish caught using hours as an exposure offset.

    Modeling strategy:
      1. Fit a Poisson GLM with offset = log_hours.
      2. Compute dispersion (Pearson Chi-square / df_resid). If dispersion > 1.5,
         fit a Negative Binomial GLM (to account for overdispersion) and return it
         as the preferred model. Otherwise return the Poisson model.

    Returns a dictionary with keys:
      - 'chosen_model': the fitted model object chosen (either 'poisson' or 'negative_binomial')
      - 'poisson_model': fitted Poisson model (statsmodels result)
      - 'negative_binomial_model': fitted NB model if fitted (else None)
      - 'dispersion': computed dispersion statistic for the Poisson fit
    """
    df = df.copy()

    # Define model predictors (exact column names from transformed df)
    predictors = ['livebait', 'camper', 'persons_c', 'child_c']
    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')

    y = df['fish_caught']
    offset = df['log_hours']

    # Fit Poisson with exposure offset (log of hours)
    poisson_model = sm.GLM(endog=y, exog=X, family=sm.families.Poisson(), offset=offset).fit()

    # Compute dispersion (Pearson chi-square / df_resid)
    pearson_resid = poisson_model.resid_pearson
    if poisson_model.df_resid is not None and poisson_model.df_resid > 0:
        dispersion = np.sum(pearson_resid ** 2) / poisson_model.df_resid
    else:
        dispersion = np.nan

    results = {
        'poisson_model': poisson_model,
        'negative_binomial_model': None,
        'dispersion': dispersion,
        'chosen_model': 'poisson'
    }

    # If evidence of overdispersion, fit Negative Binomial GLM
    # threshold chosen conservatively (1.5); adjust as needed for domain/context
    if not np.isnan(dispersion) and dispersion > 1.5:
        try:
            nb_model = sm.GLM(endog=y, exog=X, family=sm.families.NegativeBinomial(), offset=offset).fit()
            results['negative_binomial_model'] = nb_model
            results['chosen_model'] = 'negative_binomial'
        except Exception:
            # If NB fails for any reason, keep Poisson and report NB as None
            results['negative_binomial_model'] = None
            results['chosen_model'] = 'poisson'

    return results


