from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/replace_with_rvs_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the fishing dataset for modeling.

    Produces the following additional columns used in modeling:
      - total_people: persons + child
      - fish_per_hour: fish_caught / hours (informative rate)
      - log_hours: natural log of hours (used as offset in GLM)

    Drops rows with missing or invalid key values (fish_caught, hours) and rows with non-positive hours.
    Ensures binary columns are int type.
    """
    df = df.copy()

    # Drop rows with missing crucial variables
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove rows with non-positive hours (cannot compute rate / offset)
    df = df[df['hours'] > 0]

    # Ensure binary indicators are integer 0/1
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].fillna(0).astype(int)
    else:
        df['livebait'] = 0
    if 'camper' in df.columns:
        df['camper'] = df['camper'].fillna(0).astype(int)
    else:
        df['camper'] = 0

    # Fill persons/child missing with 0 if necessary (conservative)
    if 'persons' in df.columns:
        df['persons'] = df['persons'].fillna(0).astype(float)
    else:
        df['persons'] = 0.0
    if 'child' in df.columns:
        df['child'] = df['child'].fillna(0).astype(float)
    else:
        df['child'] = 0.0

    # Derived columns
    df['total_people'] = df['persons'] + df['child']

    # Rate per hour (informative summary variable)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Log hours for use as an offset in log-linear count models
    # Ensure numerical stability (hours > 0 guaranteed above)
    df['log_hours'] = np.log(df['hours'].astype(float))

    # Keep only columns necessary for modeling plus originals for diagnostics
    required_cols = [
        'fish_caught', 'fish_per_hour', 'livebait', 'camper',
        'persons', 'child', 'total_people', 'hours', 'log_hours'
    ]
    # If original df had additional columns, keep them too; but ensure required are present
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count models for fish_caught using hours as exposure.

    Steps:
      1. Fit Poisson GLM with a log link and offset = log_hours.
      2. Compute dispersion (Pearson chi-square / df_resid). If substantial overdispersion (dispersion > 1.5), fit a Negative Binomial model.

    Returns a dictionary with fitted model objects and diagnostics:
      - 'poisson': fitted Poisson GLMResults
      - 'dispersion': computed dispersion statistic
      - 'used_model': 'negative_binomial' or 'poisson'
      - 'negative_binomial': fitted NegativeBinomial (if fitted) or None

    Note: callers can inspect model.summary() on returned model objects.
    """
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial

    # Ensure required columns exist
    for col in ['fish_caught', 'log_hours', 'livebait', 'camper', 'total_people']:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # Endogenous and exogenous
    y = df['fish_caught'].astype(float)
    X = df[['livebait', 'camper', 'total_people']].astype(float)
    X = sm.add_constant(X, has_constant='add')
    offset = df['log_hours'].astype(float)

    results = {
        'poisson': None,
        'negative_binomial': None,
        'dispersion': None,
        'used_model': None
    }

    # Fit Poisson GLM with offset
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    results['poisson'] = poisson_model

    # Compute Pearson chi-square dispersion: sum((y - mu)^2 / mu) / df_resid
    mu = poisson_model.mu
    # Avoid division by zero by a tiny epsilon where mu==0
    eps = 1e-8
    pearson_chi2 = np.sum((y - mu) ** 2 / (mu + eps))
    dispersion = pearson_chi2 / poisson_model.df_resid if poisson_model.df_resid > 0 else np.nan
    results['dispersion'] = float(dispersion)

    # If overdispersion is substantial, fit Negative Binomial
    if not np.isnan(dispersion) and dispersion > 1.5:
        try:
            nb_model = NegativeBinomial(y, X, offset=offset).fit(disp=0)
            results['negative_binomial'] = nb_model
            results['used_model'] = 'negative_binomial'
        except Exception:
            # Fallback: try GLM NegativeBinomial family
            try:
                nb_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()
                results['negative_binomial'] = nb_glm
                results['used_model'] = 'negative_binomial_glm'
            except Exception:
                # If NB fails, keep Poisson
                results['used_model'] = 'poisson'
    else:
        results['used_model'] = 'poisson'

    return results


