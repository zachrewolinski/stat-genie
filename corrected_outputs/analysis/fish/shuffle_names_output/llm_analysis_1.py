from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Attempt to read example dataframe if available, but do not raise on import
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/shuffle_names_output/fish.csv')
except Exception:
    df = None

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe to the analysis-ready dataframe. Creates exposure (person-hours), a log-offset, and a simple rate column.

    Columns produced and required by the model:
      - fish_caught           (original outcome; numeric)
      - livebait              (binary predictor; coerced to int 0/1)
      - child                 (binary control; coerced to int 0/1)
      - camper                (numeric control; coerced to int)
      - persons               (numeric control; used to compute exposure)
      - hours                 (numeric; used to compute exposure)
      - ExposureHoursPersons  (persons * hours; exposure variable)
      - log_exposure          (log of ExposureHoursPersons; used as offset)
      - fish_per_hour         (descriptive: fish_caught / hours)
    """
    df = df.copy()

    # Ensure numeric types for key columns; coerce errors to NaN
    for c in ['fish_caught', 'hours', 'persons', 'livebait', 'child', 'camper']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing outcome or exposure components
    df = df.dropna(subset=['fish_caught', 'hours', 'persons'])

    # Compute exposure: person-hours of effort (persons * hours)
    df['ExposureHoursPersons'] = df['hours'] * df['persons']

    # Remove rows with nonpositive exposure (cannot define a rate/exposure of zero)
    df = df[df['ExposureHoursPersons'] > 0].copy()

    # Log exposure used as offset in GLM
    df['log_exposure'] = np.log(df['ExposureHoursPersons'])

    # Ensure binary controls/predictors are 0/1 integers where possible
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].fillna(0).astype(int)
    else:
        # If missing, create a default column (no live bait)
        df['livebait'] = 0

    if 'child' in df.columns:
        df['child'] = df['child'].fillna(0).astype(int)
    else:
        df['child'] = 0

    # camper may be numeric count (coerce to int)
    if 'camper' in df.columns:
        df['camper'] = df['camper'].fillna(0).astype(int)
    else:
        df['camper'] = 0

    # Descriptive rate per group-hour (useful for diagnostics)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Final columns check: keep only relevant columns for modeling and diagnostics
    keep_cols = ['fish_caught', 'livebait', 'child', 'camper', 'persons', 'hours', 'ExposureHoursPersons', 'log_exposure', 'fish_per_hour']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit count models to estimate fish caught rate per person-hour.

    Strategy:
      1) Fit a Poisson GLM with offset = log_exposure to estimate per-person-per-hour rate.
      2) Compute dispersion (Pearson chi2 / df_resid). If overdispersion is present (dispersion > 1.5), also fit a Negative Binomial GLM.

    Model specification (group-level total counts with exposure):
      fish_caught ~ livebait + child + camper  ,  offset = log_exposure

    Returns a dict containing fitted model result objects and diagnostics.
    """
    df = df.copy()

    # Required columns check
    required = ['fish_caught', 'livebait', 'child', 'camper', 'log_exposure']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    formula = 'fish_caught ~ livebait + child + camper'

    # Fit Poisson GLM with log-exposure offset
    poisson_model = sm.GLM.from_formula(formula, data=df, family=sm.families.Poisson(), offset=df['log_exposure'])
    poisson_results = poisson_model.fit()

    # Compute dispersion (Pearson chi-square / df_resid)
    try:
        pearson_chi2 = ((poisson_results.resid_pearson) ** 2).sum()
        dispersion = pearson_chi2 / poisson_results.df_resid if poisson_results.df_resid > 0 else np.nan
    except Exception:
        pearson_chi2 = np.nan
        dispersion = np.nan

    results = {
        'poisson_results': poisson_results,
        'poisson_aic': poisson_results.aic,
        'dispersion': dispersion
    }

    # If overdispersion, fit Negative Binomial GLM as robustness check
    if (not np.isnan(dispersion)) and (dispersion > 1.5):
        try:
            nb_model = sm.GLM.from_formula(formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_exposure'])
            nb_results = nb_model.fit()
            results['nb_results'] = nb_results
            results['nb_aic'] = nb_results.aic
        except Exception as e:
            results['nb_error'] = str(e)

    # Also provide a simple OLS on the rate (fish_per_hour) as a descriptive alternative
    if 'fish_per_hour' in df.columns:
        try:
            # log-transform to reduce skew and handle zeros by adding a tiny constant
            df['log_fph'] = np.log(df['fish_per_hour'].clip(lower=1e-6))
            ols_model = sm.OLS.from_formula('log_fph ~ livebait + child + camper', data=df).fit()
            results['ols_log_rate'] = ols_model
        except Exception as e:
            results['ols_error'] = str(e)

    return results