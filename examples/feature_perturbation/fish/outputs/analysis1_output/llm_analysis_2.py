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
    Transform raw fishing-visit data into a modelling-ready dataframe.

    Produces the following additional columns used in the model:
      - persons_total : numeric (persons + child)
      - rate_per_hour : fish_caught / hours (for descriptive checks)
      - log_hours : natural log of hours (used as offset in count models)

    Drops rows with missing or invalid values for fish_caught or hours (hours <= 0 cannot be used
    as exposure). Ensures binary columns are integer 0/1.
    """

    # copy to avoid modifying original
    df = df.copy()

    # Drop rows with missing essential variables
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove non-positive or extremely small hours values (cannot take log of 0/neg).
    # Keep a small tolerance for floating noise: require hours > 0
    df = df.loc[df['hours'] > 0].reset_index(drop=True)

    # Ensure binary columns are numeric 0/1 and handle missing values
    for col in ['livebait', 'camper']:
        if col in df.columns:
            # coerce to numeric then fillna with 0 (assume no) if desired; here drop rows with missing binary
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['livebait', 'camper'])
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Create total persons variable (adults + children). If child column missing, treat as 0.
    if 'child' not in df.columns:
        df['child'] = 0
    if 'persons' not in df.columns:
        # if persons missing, we cannot compute group size; drop such rows
        df = df.dropna(subset=['persons'])
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df = df.dropna(subset=['persons'])

    df['child'] = pd.to_numeric(df['child'], errors='coerce').fillna(0)

    df['persons_total'] = df['persons'] + df['child']

    # Descriptive rate column (not used directly in the GLM but useful)
    df['rate_per_hour'] = df['fish_caught'] / df['hours']

    # Offset: log of exposure hours
    df['log_hours'] = np.log(df['hours'])

    # Keep only columns needed downstream plus originals for inspection
    keep_cols = ['fish_caught', 'hours', 'log_hours', 'rate_per_hour', 'livebait', 'camper', 'persons', 'child', 'persons_total']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression for fish_caught with exposure (hours) as an offset to estimate
    catch rate per hour. The function fits a Poisson GLM first and evaluates overdispersion.
    If substantial overdispersion is detected (Pearson chi-square / df > 1.5), it refits
    using a Negative Binomial GLM.

    Model specification (exog): constant + livebait + camper + persons_total
    Offset: log_hours (log of hours)

    Returns the fitted results object from statsmodels (either Poisson or Negative Binomial fit).
    """

    # require necessary columns
    required = ['fish_caught', 'log_hours', 'livebait', 'camper', 'persons_total']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Transformed dataframe is missing required columns: {}'.format(missing))

    # prepare endog and exog
    endog = df['fish_caught']
    exog = df[['livebait', 'camper', 'persons_total']].astype(float)
    exog = sm.add_constant(exog, has_constant='add')
    offset = df['log_hours']

    # Fit Poisson GLM with log-link and offset
    poisson_model = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset)
    poisson_results = poisson_model.fit()

    # Compute Pearson chi2 dispersion statistic: sum((resid_pearson**2)) / df_resid
    pearson_chi2 = sum(poisson_results.resid_pearson**2)
    df_resid = poisson_results.df_resid if hasattr(poisson_results, 'df_resid') else (len(endog) - exog.shape[1])
    dispersion = pearson_chi2 / max(df_resid, 1)

    # If overdispersion detected, refit with Negative Binomial
    if dispersion > 1.5:
        try:
            nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(), offset=offset)
            nb_results = nb_model.fit()
            nb_results.model_fit_family = 'NegativeBinomial'
            nb_results.dispersion_stat = dispersion
            return nb_results
        except Exception:
            # If GLM NegativeBinomial not available/fails, return Poisson with overdispersion note
            poisson_results.model_fit_family = 'Poisson'
            poisson_results.dispersion_stat = dispersion
            return poisson_results
    else:
        poisson_results.model_fit_family = 'Poisson'
        poisson_results.dispersion_stat = dispersion
        return poisson_results


