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
    Transform the raw park fishing dataset into the analysis-ready dataframe.
    - Ensures numeric types
    - Drops rows with missing or non-positive hours or missing fish_caught
    - Creates total_people, catch_per_hour, and log_hours (for offset)
    - Ensures binary indicators are integer 0/1

    Returns the transformed dataframe containing at minimum the columns used by the model:
    ['fish_caught','livebait','camper','persons','child','total_people','hours','log_hours','catch_per_hour']
    """
    df = df.copy()

    # Ensure expected columns exist
    required = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Coerce to numeric where appropriate
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows without fish count or hours
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove zero or negative hours (can't compute rate / offset)
    df = df[df['hours'] > 0]

    # Create derived variables
    df['total_people'] = df['persons'].fillna(0) + df['child'].fillna(0)

    # Descriptive rate (not used directly as DV in GLM with offset)
    df['catch_per_hour'] = df['fish_caught'] / df['hours']

    # Log of hours for use as offset in count regression
    df['log_hours'] = np.log(df['hours'])

    # Ensure binary covariates are ints (0/1). Coerce non-binary values to 0/1 by rounding if necessary.
    df['livebait'] = df['livebait'].fillna(0).astype(int)
    df['camper'] = df['camper'].fillna(0).astype(int)

    # Final minimal column set check (for clarity)
    keep_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'total_people', 'hours', 'log_hours', 'catch_per_hour']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count regression for fish_caught with exposure-hours as an offset to estimate fish-per-hour rate.

    Procedure:
    1. Fit a Poisson GLM with offset=log_hours.
    2. Compute overdispersion (Pearson chi-square / df_resid).
    3. If overdispersion > 1.5, refit with a Negative Binomial GLM.

    Returns a dictionary with:
      - 'model_type': 'Poisson' or 'NegativeBinomial'
      - 'final_results': fitted results object (statsmodels result)
      - 'poisson_results': poisson fit (always present)
      - 'dispersion': computed dispersion statistic
    """
    # Copy to avoid modifying the caller's dataframe
    df = df.copy()

    # Verify required columns
    for col in ['fish_caught', 'log_hours']:
        if col not in df.columns:
            raise ValueError(f"Required column for modeling missing: {col}")

    # Define formula: predict total fish counts with covariates; hours enters as an offset
    formula = 'fish_caught ~ livebait + camper + total_people + child'

    # Fit Poisson GLM with offset = log_hours
    poisson_model = sm.GLM.from_formula(formula, data=df, family=sm.families.Poisson(), offset=df['log_hours'])
    poisson_results = poisson_model.fit()

    # Overdispersion check: Pearson chi2 / df_resid
    pearson_chi2 = np.sum(poisson_results.resid_pearson ** 2)
    dispersion = pearson_chi2 / poisson_results.df_resid if poisson_results.df_resid > 0 else np.nan

    # Threshold for switching to Negative Binomial (common rule-of-thumb)
    if not np.isnan(dispersion) and dispersion > 1.5:
        nb_model = sm.GLM.from_formula(formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_hours'])
        nb_results = nb_model.fit()
        final_results = nb_results
        model_type = 'NegativeBinomial'
    else:
        final_results = poisson_results
        model_type = 'Poisson'

    # Return useful objects for inspection (the fitted results objects contain .summary())
    return {
        'model_type': model_type,
        'final_results': final_results,
        'poisson_results': poisson_results,
        'dispersion': dispersion
    }


