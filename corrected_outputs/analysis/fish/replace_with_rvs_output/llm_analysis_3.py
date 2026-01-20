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
    Transform the raw fishing dataset into the dataframe used for modeling.
    Produces derived columns: group_size, fish_per_hour, log_hours.

    Inputs (expected columns in original df):
      - fish_caught, livebait, camper, persons, child, hours

    Returns: dataframe with the original columns plus the derived columns.
    """
    df = df.copy()

    # Ensure expected columns exist
    expected_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input dataframe: {missing}")

    # Convert to numeric where appropriate; coerce errors to NaN
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce')
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce')
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df['child'] = pd.to_numeric(df['child'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')

    # Drop rows with missing values in any of the analytic columns
    df = df.dropna(subset=expected_cols)

    # Ensure hours are positive; replace non-positive with a small epsilon to allow offset
    # (Dataset min is >0 in provided schema, but this guards against zeros).
    eps = 1e-3
    df.loc[df['hours'] <= 0, 'hours'] = eps

    # Derived variables
    df['group_size'] = df['persons'] + df['child']

    # Descriptive rate variable (useful for plotting / checks)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Offset used in GLM should be the log of hours
    df['log_hours'] = np.log(df['hours'])

    # Ensure binary variables are 0/1 integers
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Optionally, you might filter out extreme outliers in hours or fish_caught here.
    # For transparency we keep all cleaned rows.

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count-regression (rate) model for fish caught per visit, using hours as exposure.
    Steps:
      1. Fit a Poisson GLM with a log link and log(hours) as offset.
      2. Compute Pearson dispersion to check overdispersion.
      3. If substantial overdispersion is detected, fit a Negative Binomial GLM and choose it.

    Returns a dictionary with the fitted models, chosen model, and diagnostics.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'log_hours']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Define formula for predictors. This models the total count while using hours as exposure (offset).
    formula = 'fish_caught ~ livebait + camper + persons + child'

    # Fit Poisson GLM with offset = log(hours)
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_hours']).fit()

    # Compute Pearson dispersion statistic: sum(pearson_resid^2) / df_resid
    pearson_resid = poisson_model.resid_pearson
    dispersion = np.sum(pearson_resid**2) / poisson_model.df_resid

    nb_model = None
    chosen = poisson_model

    # Heuristic: if dispersion substantially > 1, consider Negative Binomial
    if dispersion > 1.5:
        try:
            # Fit Negative Binomial GLM (log link, offset). This addresses overdispersion.
            nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_hours']).fit()

            # Choose NB if its AIC is lower than Poisson's AIC (indicating better fit accounting for complexity)
            if nb_model.aic < poisson_model.aic:
                chosen = nb_model
        except Exception:
            # If NB fit fails, keep Poisson but return a note in diagnostics
            nb_model = None

    results = {
        'chosen_model': chosen,
        'poisson_model': poisson_model,
        'negative_binomial_model': nb_model,
        'dispersion': dispersion,
        'formula': formula
    }

    return results


