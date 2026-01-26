from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/negative_leading_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling-ready dataframe.

    Produces these columns required by the model:
      - efficiency: nuts_opened / seconds
      - log_efficiency: log(efficiency + eps)
      - age_c: centered age
      - sex_M: 1 for male, 0 for female
      - help_Y: 1 if help received, 0 otherwise
      - hammer: categorical (kept as-is but converted to str)
      - chimpanzee: grouping id (kept as-is)

    Removes rows with missing or invalid data for these columns.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing core values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Remove rows with non-positive session duration to avoid division by zero or negative rates
    df = df[df['seconds'] > 0]

    # Compute raw efficiency (nuts per second)
    df['efficiency'] = df['nuts_opened'].astype(float) / df['seconds'].astype(float)

    # Remove infinities / NaNs if any
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['efficiency'])

    # Log-transform the efficiency to stabilize variance and reduce skew
    eps = 1e-8
    df['log_efficiency'] = np.log(df['efficiency'] + eps)

    # Center age for interpretability
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()

    # Encode sex: 1 for male, 0 for female (case-insensitive)
    df['sex_M'] = df['sex'].astype(str).str.strip().str.lower().map(lambda x: 1 if x == 'm' else 0)

    # Encode help: accept 'y', 'yes' (case-insensitive) as positive; everything else -> 0
    df['help_Y'] = df['help'].astype(str).str.strip().str.lower().map(lambda x: 1 if x.startswith('y') else 0)

    # Ensure hammer is a string/categorical column
    df['hammer'] = df['hammer'].astype(str)

    # Ensure chimpanzee id is preserved for grouping
    # (make sure it's categorical/int as appropriate; keep original type)
    df['chimpanzee'] = df['chimpanzee']

    # Final sanity: drop rows where log_efficiency is NaN (shouldn't happen but safety)
    df = df.dropna(subset=['log_efficiency', 'age_c', 'sex_M', 'help_Y', 'hammer', 'chimpanzee'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit models testing whether age, sex, and receiving help influence nut-cracking efficiency.

    Models fit:
      1) Linear mixed-effects model (random intercept for chimpanzee):
            log_efficiency ~ age_c + sex_M + help_Y + C(hammer)  (groups=chimpanzee)
         - This is the preferred specification because data are nested within individuals.

      2) OLS with cluster-robust standard errors clustered by chimpanzee (robustness check):
            log_efficiency ~ age_c + sex_M + help_Y + C(hammer)

    Returns a dict with fitted model objects (or error strings if fitting failed).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Required columns check
    for col in ['log_efficiency', 'age_c', 'sex_M', 'help_Y', 'hammer', 'chimpanzee']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe. Please run transform() first.")

    # Fit mixed-effects model with random intercept for chimpanzee
    formula = 'log_efficiency ~ age_c + sex_M + help_Y + C(hammer)'
    try:
        md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
        # Use default optimization; if convergence issues arise, try different methods
        mdf = md.fit(reml=False)
        results['mixedlm'] = mdf
    except Exception as e:
        # Capture the exception text so user knows what failed
        results['mixedlm_error'] = str(e)

    # Fit OLS with cluster-robust standard errors (clustered by chimpanzee)
    try:
        ols = smf.ols(formula, data=df).fit()
        # Compute cluster-robust covariance (cluster by chimpanzee)
        try:
            ols_cluster = ols.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
            results['ols_cluster'] = ols_cluster
        except Exception:
            # If cluster robust fails for some reason, still return the plain OLS fit
            results['ols'] = ols
    except Exception as e:
        results['ols_error'] = str(e)

    # Return the results dictionary so the caller can inspect summaries, params, etc.
    return results


