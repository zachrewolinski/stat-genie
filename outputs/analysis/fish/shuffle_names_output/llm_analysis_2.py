from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/shuffle_names_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fishing-trip dataframe into the analytic dataframe.
    Creates:
      - CatchRatePerHour: fish_caught / hours
      - LogCatchRate: natural log of CatchRatePerHour (small constant added to avoid log(0))
      - log_persons: log(persons + 1) to reduce skew and capture diminishing returns
    Ensures binary variables are integer 0/1 and filters out invalid rows (missing key columns or non-positive hours).
    Returns the dataframe containing at least the columns used in the model: ['CatchRatePerHour','LogCatchRate','log_persons','livebait','child','camper','hours','fish_caught','persons']
    """

    # Copy to avoid modifying original df in-place
    df = df.copy()

    # Required raw columns
    required_cols = ['fish_caught', 'hours', 'persons', 'livebait', 'child', 'camper']
    # Drop rows with missing critical values
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where appropriate
    for col in ['fish_caught', 'hours', 'persons', 'livebait', 'child', 'camper']:
        # coerce invalid entries to NaN then drop
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required_cols)

    # Remove rows with non-positive hours (can't compute a meaningful per-hour rate)
    df = df[df['hours'] > 0]

    # Compute catch rate per hour
    df['CatchRatePerHour'] = df['fish_caught'] / df['hours']

    # Replace infinite or extremely large values if any (safeguard)
    df.loc[np.isinf(df['CatchRatePerHour']), 'CatchRatePerHour'] = np.nan
    df = df.dropna(subset=['CatchRatePerHour'])

    # Small constant to avoid log(0). Using a value much smaller than the typical minimal observed rate
    eps = 1e-6
    df['LogCatchRate'] = np.log(df['CatchRatePerHour'] + eps)

    # Log-transform of group size (persons) to capture diminishing returns; add 1 to handle zero-person groups if present
    df['log_persons'] = np.log(df['persons'] + 1)

    # Ensure binary predictors are 0/1 integers
    df['livebait'] = df['livebait'].astype(int)
    df['child'] = df['child'].astype(int)

    # Ensure camper is numeric integer (count of campers)
    df['camper'] = df['camper'].astype(int)

    # Optional: remove extremely large outliers in CatchRatePerHour (e.g., > 99.9th percentile) to reduce influence
    # Comment/uncomment as needed. Here we winsorize at 99.9 percentile.
    try:
        upper = df['CatchRatePerHour'].quantile(0.999)
        df.loc[df['CatchRatePerHour'] > upper, 'CatchRatePerHour'] = upper
        df['LogCatchRate'] = np.log(df['CatchRatePerHour'] + eps)
    except Exception:
        pass

    # Final columns required for the model
    final_cols = ['CatchRatePerHour', 'LogCatchRate', 'log_persons', 'livebait', 'child', 'camper', 'hours', 'fish_caught', 'persons']
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a regression model predicting log(catch rate per hour).
    Model: OLS on LogCatchRate with predictors log_persons, livebait, child, camper, and an interaction between livebait and log_persons.
    Uses heteroskedasticity-robust (HC3) standard errors.

    Returns the fitted model results object (statsmodels regression results).
    """

    # local import for formula API
    import statsmodels.formula.api as smf

    # Drop rows with missing values in model columns
    model_df = df.dropna(subset=['LogCatchRate', 'log_persons', 'livebait', 'child', 'camper'])

    # Define formula. Interaction allows the effect of group size to differ by livebait use.
    formula = 'LogCatchRate ~ log_persons + livebait + child + camper + livebait:log_persons'

    # Fit OLS on the log-rate
    results = smf.ols(formula=formula, data=model_df).fit(cov_type='HC3')

    # Print a concise summary for quick inspection (callers can use returned object for further inspection)
    print(results.summary())

    return results


