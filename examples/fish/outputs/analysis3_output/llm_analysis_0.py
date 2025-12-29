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
    Transform the raw fishing-visit dataframe into a modeling-ready dataframe.

    Adds the following columns used by the models:
      - fish_per_hour: fish_caught / hours
      - log_fish_per_hour: log((fish_caught + 0.1) / hours) (small offset to handle zeros)
      - HasCamper: binary indicator (1 if camper > 0, else 0)
    Ensures binary columns are integers and removes rows with missing or invalid hours.
    """
    df = df.copy()

    # Required columns
    required_cols = ['fish_caught', 'hours', 'persons', 'livebait', 'camper', 'child']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing fish or hours or persons
    df = df.dropna(subset=['fish_caught', 'hours', 'persons'])

    # Keep only rows with positive hours (cannot compute rate otherwise)
    df = df[df['hours'] > 0].copy()

    # Ensure numeric types
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')

    # After coercion drop any newly introduced NA values
    df = df.dropna(subset=['fish_caught', 'hours', 'persons'])

    # Create fish per hour (rate)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Create a logged version for OLS modeling. Add a small constant to fish_caught to handle zeros.
    # The constant (0.1) is a small pseudo-count; adjust if domain knowledge suggests a different value.
    df['log_fish_per_hour'] = np.log((df['fish_caught'] + 0.1) / df['hours'])

    # Coerce binary predictors to integers (0/1)
    # If the data encodes child/livebait as 0/1 already this is safe; otherwise non-zero -> 1
    df['livebait'] = df['livebait'].fillna(0).astype(int)
    # child may be 0/1; coerce similarly
    df['child'] = df['child'].fillna(0).astype(int)

    # Derive HasCamper as binary indicator from camper count (0 means no camper)
    # Keep original camper count as well in case it's useful
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce').fillna(0)
    df['HasCamper'] = (df['camper'] > 0).astype(int)

    # Final sanity: drop any rows with infinite or NaN values in created columns
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['fish_per_hour', 'log_fish_per_hour'])

    # Return the dataframe containing all columns needed for modeling
    # (we leave other original columns intact, but ensure required derived ones exist)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to estimate factors associated with fish caught per hour:
      1) OLS on log(fish_per_hour) (linearized multiplicative model) with robust SEs.
      2) Poisson log-rate GLM for fish_caught using hours as an offset (estimates multiplicative effects on rate).

    Returns a dictionary with fitted results objects: {'ols': ols_results, 'poisson': poisson_results_robust}
    """
    # Ensure required columns exist
    required = ['log_fish_per_hour', 'fish_caught', 'hours', 'livebait', 'HasCamper', 'persons', 'child']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modelling: {missing}")

    # Define predictors
    predictors = ['livebait', 'HasCamper', 'persons', 'child']

    # Build design matrix
    X = df[predictors].copy()
    X = sm.add_constant(X)

    # 1) OLS on log(fish_per_hour)
    y_ols = df['log_fish_per_hour']
    ols_model = sm.OLS(y_ols, X)
    ols_results = ols_model.fit(cov_type='HC3')  # robust (heteroskedasticity-consistent) SEs

    # 2) Poisson GLM for counts with log(hours) as offset (models rate = exp(X*beta) per hour)
    # Note: poisson expects the dependent to be non-negative. fish_caught may be non-integer; GLM allows floats.
    poisson_model = sm.GLM(df['fish_caught'], X, family=sm.families.Poisson(), offset=np.log(df['hours']))
    poisson_res = poisson_model.fit()
    # Obtain robust covariance estimates for the GLM results
    try:
        poisson_res_robust = poisson_res.get_robustcov_results(cov_type='HC3')
    except Exception:
        # Fallback: if robustcov not available, use the original result
        poisson_res_robust = poisson_res

    # Print concise summaries for quick inspection (can be removed in production)
    print('OLS on log(fish_per_hour) summary:')
    print(ols_results.summary())
    print('\nPoisson GLM (hours as offset) robust-summary:')
    print(poisson_res_robust.summary())

    # Return results so caller code can further inspect coefficients, conf intervals, predicted rates, etc.
    return {
        'ols': ols_results,
        'poisson': poisson_res_robust
    }


