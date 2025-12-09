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
    Transform the raw fishing-visits dataframe into the modeling dataframe.

    - Drops rows with missing fish_caught or hours and removes non-positive hours (can't take log of 0).
    - Ensures binary columns are integer 0/1.
    - Builds total_people and a mean-centered version total_people_c.
    - Computes fish_per_hour for descriptive checks and log_hours for use as an offset in GLM.

    Returns the dataframe containing at least the following columns (used in modeling):
      ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'total_people', 'total_people_c', 'hours', 'log_hours', 'fish_per_hour']
    """
    df = df.copy()

    # Drop rows missing essential variables
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove rows with non-positive hours (cannot use as exposure)
    df = df[df['hours'] > 0].copy()

    # Ensure binary fields are integers (0/1)
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].astype(int)
    else:
        df['livebait'] = 0

    if 'camper' in df.columns:
        df['camper'] = df['camper'].astype(int)
    else:
        df['camper'] = 0

    # Ensure numeric columns for group composition
    if 'persons' in df.columns:
        df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    else:
        df['persons'] = 1
    if 'child' in df.columns:
        df['child'] = pd.to_numeric(df['child'], errors='coerce')
    else:
        df['child'] = 0

    # Fill missing reasonable defaults (if any) and keep as numeric
    df['persons'] = df['persons'].fillna(1)
    df['child'] = df['child'].fillna(0)

    # Derived variables
    df['total_people'] = df['persons'] + df['child']
    df['total_people_c'] = df['total_people'] - df['total_people'].mean()

    # Descriptive rate and offset
    df['fish_per_hour'] = df['fish_caught'] / df['hours']
    df['log_hours'] = np.log(df['hours'])

    # Keep only columns necessary for modeling and interpretation
    keep_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'total_people', 'total_people_c', 'hours', 'log_hours', 'fish_per_hour']
    # If extra columns are missing in the input, ensure they exist in the output (already handled above)
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression for fish_caught with hours as exposure (offset).

    Procedure:
    1. Fit Poisson GLM with offset=log_hours and predictors: livebait, camper, total_people_c.
    2. Compute dispersion using the Pearson chi-square / df_resid.
    3. If dispersion > 1.5 (substantial overdispersion), fit a Negative Binomial GLM instead and return it as the final model.

    Returns a dictionary with keys:
      - 'final_model': the fitted statsmodels results object (Poisson or NegativeBinomial)
      - 'model_type': 'Poisson' or 'NegativeBinomial'
      - 'dispersion': dispersion statistic computed from the Poisson fit
      - 'poisson_results': the Poisson fit object (useful for diagnostics)
    """
    df = df.copy()

    # Response and predictors
    y = df['fish_caught'].astype(float)
    X = df[['livebait', 'camper', 'total_people_c']]
    X = sm.add_constant(X, has_constant='add')

    # Poisson GLM with log-hours offset
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=df['log_hours'])
    poisson_results = poisson_model.fit()

    # Predicted mean under Poisson
    mu = poisson_results.predict()

    # Pearson chi-square for dispersion: sum((y - mu)^2 / mu)
    # guard against zero mu
    mu_safe = np.where(mu <= 0, 1e-8, mu)
    pearson_chi2 = np.sum((y - mu_safe) ** 2 / mu_safe)
    df_resid = poisson_results.df_resid if hasattr(poisson_results, 'df_resid') else (len(y) - poisson_results.params.size)
    dispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan

    final_results = poisson_results
    model_type = 'Poisson'

    # If overdispersed, fit Negative Binomial GLM (NB2 in statsmodels)
    if not np.isnan(dispersion) and dispersion > 1.5:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_hours'])
        # alpha initial value is a starting point; statsmodels will estimate it for some implementations
        nb_results = nb_model.fit()
        final_results = nb_results
        model_type = 'NegativeBinomial'

    return {
        'final_model': final_results,
        'model_type': model_type,
        'dispersion': dispersion,
        'poisson_results': poisson_results
    }


