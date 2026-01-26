from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/shuffle_names_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into an analysis-ready dataframe.

    Key steps/assumptions (based on provided schema where some column descriptions were swapped):
    - 'seconds' column contains individual ID -> rename to 'ID'
    - 'nuts_opened' column actually contains the individual's age in years -> rename to 'age_years'
    - 'age' column contains sex ('f'/'m') -> rename to 'sex'
    - 'help' column contains the number of nuts opened in the session -> rename to 'nuts_opened'
    - 'sex' column (numeric in raw schema) contains session duration in seconds -> rename to 'session_seconds'
    - 'chimpanzee' column indicates whether focal received help -> rename to 'received_help' (values like 'y'/'N')

    Derived columns:
    - nuts_per_sec: nuts_opened / session_seconds
    - log_nps: log(nuts_per_sec + small_constant) used as DV in model

    The function returns a dataframe containing at minimum the columns referenced in the model:
    ['ID', 'age_years', 'sex', 'received_help_bin', 'hammer', 'nuts_opened', 'session_seconds', 'nuts_per_sec', 'log_nps']
    """
    df = df.copy()

    # Rename based on the schema's swapped descriptions
    rename_map = {
        'seconds': 'ID',                # ID of individual
        'nuts_opened': 'age_years',      # actually age in years
        'age': 'sex',                    # actually sex encoded 'f'/'m'
        'help': 'nuts_opened',           # actually number of nuts opened
        'sex': 'session_seconds',        # actually duration (seconds)
        'chimpanzee': 'received_help'    # received help from another chimp (y/N)
    }
    # Only rename the columns that exist
    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_map)

    # Ensure necessary columns exist; if not, create placeholders to avoid KeyErrors later (they will be dropped)
    required_columns = ['ID', 'age_years', 'sex', 'nuts_opened', 'session_seconds', 'received_help', 'hammer']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Clean and coerce types

    # age_years: numeric
    df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')

    # sex: normalize to 'M'/'F' categorical
    df['sex'] = df['sex'].astype(str).str.strip().str.lower().replace({'m': 'M', 'f': 'F', 'male': 'M', 'female': 'F'})
    # Any values not 'M' or 'F' -> missing
    df.loc[~df['sex'].isin(['M', 'F']), 'sex'] = pd.NA
    df['sex'] = df['sex'].astype('category')

    # nuts_opened: numeric (number of nuts opened in session)
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')

    # session_seconds: numeric (duration of session in seconds)
    df['session_seconds'] = pd.to_numeric(df['session_seconds'], errors='coerce')

    # received_help: map to binary 1/0. Accept many representations.
    def map_received_help(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip().lower()
        if s in ['y', 'yes', '1', 'true', 't']:
            return 1
        if s in ['n', 'no', '0', 'false', 'f']:
            return 0
        # keep NA for anything else
        return pd.NA

    df['received_help_bin'] = df['received_help'].apply(map_received_help)

    # hammer: keep as categorical; strip whitespace
    if 'hammer' in df.columns:
        # convert to string first to unify representations like np.nan -> 'nan', then replace that placeholder with actual NA
        df['hammer'] = df['hammer'].astype(str).str.strip().replace({'nan': pd.NA})
        df['hammer'] = df['hammer'].astype('category')
    else:
        df['hammer'] = pd.NA
        df['hammer'] = df['hammer'].astype('category')

    # Compute nuts per second (efficiency); drop or mark invalid rows where session_seconds <= 0 or missing
    df['nuts_per_sec'] = np.nan
    valid_mask = df['session_seconds'].notna() & (df['session_seconds'] > 0) & df['nuts_opened'].notna()
    df.loc[valid_mask, 'nuts_per_sec'] = df.loc[valid_mask, 'nuts_opened'] / df.loc[valid_mask, 'session_seconds']

    # Remove rows with missing essential data for the planned analysis
    # Include ID because the model uses it as a grouping variable and it must be present
    essential = ['nuts_per_sec', 'age_years', 'sex', 'received_help_bin', 'ID']
    # Before dropping, ensure that pure-missing ID placeholders (np.nan) are recognized as NaN (they are)
    df = df.dropna(subset=essential)

    # At this point, received_help_bin should no longer contain pd.NA; convert to a standard numpy integer dtype
    # Use to_numeric first to handle any stray values, then cast to int64
    df['received_help_bin'] = pd.to_numeric(df['received_help_bin'], errors='coerce').astype('int64')

    # Now ensure ID is stored as string (object) for grouping
    df['ID'] = df['ID'].astype(str)

    # Log-transform the efficiency to stabilize skew. Add small constant to avoid log(0).
    small_const = 1e-6
    df['log_nps'] = np.log(df['nuts_per_sec'] + small_const)

    # Final columns used in modeling (keep extras for diagnostics)
    final_cols = ['ID', 'age_years', 'sex', 'received_help_bin', 'hammer', 'nuts_opened', 'session_seconds', 'nuts_per_sec', 'log_nps']
    for c in final_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects linear model predicting log(nuts_per_sec) from age, sex, and received_help,
    controlling for hammer type and including a random intercept for individual (ID).

    If MixedLM fails to converge or is inappropriate, fall back to OLS with clustered (by ID) robust SEs.

    Returns the fitted model object. For MixedLM this is the result returned by statsmodels (MixedLMResults).
    For the OLS fallback, returns the statsmodels RegressionResultsWrapper.
    """
    df = df.copy()

    # Ensure categorical variables are properly typed
    df['sex'] = df['sex'].astype('category')
    df['hammer'] = df['hammer'].astype('category')

    # Ensure received_help_bin is a standard numeric dtype (numpy int) for patsy/statsmodels
    # If transform was used, this should already be int64; coerce defensively
    df['received_help_bin'] = pd.to_numeric(df['received_help_bin'], errors='coerce')
    if df['received_help_bin'].isnull().any():
        # If any missing after coercion, raise a clearer error
        raise ValueError("received_help_bin contains missing or non-numeric values. Ensure transform() produced a clean dataframe.")
    df['received_help_bin'] = df['received_help_bin'].astype('int64')

    # Formula: main effects of age, sex, received_help; control for hammer
    formula = 'log_nps ~ age_years + C(sex) + received_help_bin + C(hammer)'

    # Try mixed-effects model with random intercept per ID
    try:
        md = sm.MixedLM.from_formula(formula, groups='ID', data=df)
        mdf = md.fit(reml=False, method='lbfgs', skip_hessian=True)
        # If converged, return mixed model results
        if hasattr(mdf, 'converged') and mdf.converged:
            return mdf
        # If not converged, fall back
    except Exception as e:
        # store exception for debugging but continue to OLS fallback
        _mixed_err = e

    # Fallback: OLS with cluster-robust standard errors (clustered by ID)
    ols_model = smf.ols(formula, data=df).fit()
    try:
        # get cluster-robust cov: clustered by ID
        cluster_ids = df['ID']
        robust = ols_model.get_robustcov_results(cov_type='cluster', groups=cluster_ids)
        return robust
    except Exception:
        # if clustering fails, return the plain OLS fit
        return ols_model