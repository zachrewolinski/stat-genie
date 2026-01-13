from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/noperturb_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe to produce all variables required for modeling.
    Returns a dataframe containing at least the following columns (used in the model and diagnostics):
      - fish_caught (original)
      - livebait (original)
      - camper (original)
      - persons (original)
      - child (original)
      - hours (original, with small-flooring for zeros)
      - group_size (derived)
      - pct_children (derived)
      - log_hours (derived, for use as offset)
      - rate_per_hour (derived, fish_caught / hours, descriptive)

    The function drops rows missing any of the needed original columns.
    """
    # Keep a copy to avoid modifying input in-place unexpectedly
    df = df.copy()

    # Required columns
    required_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']

    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    for col in required_cols:
        # coerce to numeric if possible
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required_cols)

    # Floor any nonpositive hours to a small positive value to allow log transform
    # (there are very small recorded hours in the data; replace 0 or negative with 0.001 hours)
    df.loc[df['hours'] <= 0, 'hours'] = 0.001

    # Derived variables
    df['group_size'] = df['persons'] + df['child']

    # Avoid division by zero in pct_children; if group_size is zero (shouldn't happen), set pct_children to 0
    df['pct_children'] = df['child'] / df['group_size']
    df.loc[df['group_size'] == 0, 'pct_children'] = 0.0

    # Log hours for offset in GLM
    df['log_hours'] = np.log(df['hours'].astype(float))

    # Descriptive derived metric: fish per hour
    df['rate_per_hour'] = df['fish_caught'] / df['hours']

    # Optionally, cast binary columns to integers (0/1)
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Final: drop any remaining rows with NA in derived cols
    df = df.dropna(subset=['group_size', 'pct_children', 'log_hours', 'rate_per_hour'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression to estimate how many fish are caught per unit time and which factors influence the catch rate.

    Procedure:
    1. Fit a Poisson GLM with an offset = log_hours (so the model estimates fish-per-hour rates).
    2. Compute the Pearson chi-square dispersion statistic. If the model is overdispersed (dispersion > 1.5), refit using a Negative Binomial family.
    3. Return a dictionary containing the chosen model results object, the Poisson results (for comparison), the dispersion statistic, and whether NB was used.

    Model formula: fish_caught ~ livebait + camper + group_size + pct_children
    Offset: log_hours
    """
    import statsmodels.api as sm

    # Ensure required model columns exist
    model_cols = ['fish_caught', 'livebait', 'camper', 'group_size', 'pct_children', 'log_hours']
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula for the GLM
    formula = 'fish_caught ~ livebait + camper + group_size + pct_children'

    # Fit Poisson with offset
    poisson_model = sm.GLM.from_formula(formula, data=df, family=sm.families.Poisson(),
                                        offset=df['log_hours'])
    poisson_results = poisson_model.fit()

    # Compute Pearson chi-square dispersion: sum(((y - mu)^2 / mu)) / df_resid
    mu = poisson_results.fittedvalues
    y = df['fish_caught'].astype(float)
    # Avoid division by zero in mu (shouldn't happen for Poisson mean) by flooring tiny values
    mu_safe = np.where(mu <= 1e-8, 1e-8, mu)
    pearson_chi2 = np.sum(((y - mu_safe) ** 2) / mu_safe)
    df_resid = poisson_results.df_resid if poisson_results.df_resid is not None and poisson_results.df_resid > 0 else max(len(y) - poisson_results.df_model - 1, 1)
    dispersion = pearson_chi2 / df_resid

    use_negative_binomial = False
    nb_results = None

    # If dispersion substantially > 1, fit Negative Binomial
    if dispersion > 1.5:
        use_negative_binomial = True
        nb_model = sm.GLM.from_formula(formula, data=df, family=sm.families.NegativeBinomial(),
                                       offset=df['log_hours'])
        try:
            nb_results = nb_model.fit()
        except Exception as e:
            # If negative binomial fails to converge, keep poisson and report the error
            nb_results = None
            use_negative_binomial = False
            # Could optionally log the exception; for now include in returned object
            nb_error = str(e)
        else:
            nb_error = None
    else:
        nb_error = None

    # Prepare results dictionary
    results = {
        'poisson_results': poisson_results,
        'dispersion': float(dispersion),
        'df_resid': float(df_resid),
        'use_negative_binomial': bool(use_negative_binomial),
        'nb_results': nb_results,
        'nb_error': nb_error,
        'formula': formula,
        'offset_column': 'log_hours'
    }

    # Print brief summaries for the user (not required, but helpful)
    print('\nModel formula:', formula)
    print('Observation count:', int(len(df)))
    print('Poisson dispersion (Pearson chi2 / df_resid):', round(dispersion, 3))
    if use_negative_binomial and nb_results is not None:
        print('\nOverdispersion detected; fitted Negative Binomial model.\n')
        print(nb_results.summary())
    else:
        print('\nUsing Poisson model (no strong overdispersion detected or NB failed).\n')
        print(poisson_results.summary())
        if nb_error is not None:
            print('\nNegative Binomial fitting error:', nb_error)

    return results


