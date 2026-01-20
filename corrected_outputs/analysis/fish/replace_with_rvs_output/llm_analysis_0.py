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
    Transform the raw fishing dataset into a dataframe ready for count-rate modeling.

    Steps:
    - Drop rows missing required columns for modeling.
    - Remove visits with non-positive hours (cannot use log(hours) as offset).
    - Ensure binary columns are integer-coded (0/1).
    - Create group_size = persons + child.
    - Create fish_rate = fish_caught / hours (useful for diagnostics/EDA; not used as model outcome).

    Returned dataframe contains at minimum the columns: ['fish_caught','livebait','camper','persons','child','hours','group_size','fish_rate']
    """
    df = df.copy()

    # Drop rows missing any key column used to build variables or used in the model
    required_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    df = df.dropna(subset=required_cols)

    # Remove rows with non-positive hours because offset requires positive exposure
    df = df[df['hours'] > 0].copy()

    # Ensure binary indicators are integer 0/1
    try:
        df['livebait'] = df['livebait'].astype(int)
    except Exception:
        # fallback: coerce boolean-like or floats to 0/1
        df['livebait'] = df['livebait'].astype(float).round().astype(int)
    try:
        df['camper'] = df['camper'].astype(int)
    except Exception:
        df['camper'] = df['camper'].astype(float).round().astype(int)

    # Derive group size and an observed rate per hour for diagnostics
    df['group_size'] = df['persons'] + df['child']

    # Avoid division by zero (hours already filtered to > 0)
    df['fish_rate'] = df['fish_caught'] / df['hours']

    # Keep only the columns needed for modeling and diagnostics (preserve original columns too)
    # The model code expects: fish_caught, livebait, camper, group_size, hours
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count regression (rate model) for fish caught using exposure (hours) as an offset.

    Approach:
    - Fit a Poisson GLM with log link and offset = log(hours).
    - Compute Pearson dispersion statistic = sum(resid_pearson^2) / df_resid. If dispersion > 1.5,
      fit a Negative Binomial GLM (to account for overdispersion) and select it as the primary model.
    - Predictors: livebait, camper, group_size (plus constant). Response: fish_caught. Offset: log(hours).

    Returns a dict containing:
      - 'poisson': fitted Poisson results (statsmodels object),
      - 'dispersion': estimated dispersion from Poisson residuals,
      - 'negative_binomial': fitted Negative Binomial results (if fitted),
      - 'chosen_model': the model chosen based on dispersion (statsmodels results object).
    """
    import numpy as _np
    import statsmodels.api as _sm

    df = df.copy()

    # Ensure columns required for modeling exist
    for col in ['fish_caught', 'livebait', 'camper', 'group_size', 'hours']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")

    # Define predictors and design matrix
    predictors = ['livebait', 'camper', 'group_size']
    X = df[predictors]
    X = _sm.add_constant(X, has_constant='add')

    # Offset is log(hours)
    offset = _np.log(df['hours'].astype(float))

    # Fit Poisson GLM
    poisson_model = _sm.GLM(df['fish_caught'], X, family=_sm.families.Poisson(), offset=offset).fit()

    # Compute Pearson dispersion = sum(resid_pearson^2) / df_resid
    pearson_chi2 = _np.sum(poisson_model.resid_pearson ** 2)
    df_resid = poisson_model.df_resid if hasattr(poisson_model, 'df_resid') else max(1, len(df) - X.shape[1])
    dispersion = pearson_chi2 / df_resid if df_resid > 0 else _np.nan

    results = {
        'poisson': poisson_model,
        'dispersion': dispersion
    }

    # If overdispersed (heuristic threshold), fit Negative Binomial
    if (dispersion is not None) and (not _np.isnan(dispersion)) and (dispersion > 1.5):
        try:
            negbin_model = _sm.GLM(df['fish_caught'], X, family=_sm.families.NegativeBinomial(), offset=offset).fit()
            results['negative_binomial'] = negbin_model
            results['chosen_model'] = negbin_model
        except Exception:
            # If NegativeBinomial fit fails, keep Poisson as fallback
            results['chosen_model'] = poisson_model
    else:
        results['chosen_model'] = poisson_model

    return results


