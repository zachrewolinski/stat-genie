from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
import warnings
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/positive_leading_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling nut-cracking efficiency.
    Produces:
      - Efficiency: nuts opened per second (float)
      - LogEfficiency: log1p(Efficiency) to stabilize skew and handle zeros
      - Sex_Male: binary 1 if sex is male, 0 if female
      - Help_Received: binary 1 if help indicates yes, 0 otherwise
      - Ensures chimpanzee is a categorical/grouping column and hammer stays as categorical
    Drops rows with missing critical values or invalid session durations (seconds <= 0).
    """
    # copy to avoid modifying original
    df = df.copy()

    # Standardize column names if necessary (expecting given schema names)
    required_cols = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Convert numeric-looking fields to numeric, coerce errors to NaN
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows with missing critical fields
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee', 'hammer'])

    # Ensure seconds > 0 to compute rates
    df = df[df['seconds'] > 0].copy()

    # Compute efficiency: nuts opened per second
    df['Efficiency'] = df['nuts_opened'].astype(float) / df['seconds'].astype(float)

    # Handle infinite or NaN values (e.g., division issues)
    df.loc[~np.isfinite(df['Efficiency']), 'Efficiency'] = np.nan
    df = df.dropna(subset=['Efficiency'])

    # Stabilize skew with log(1 + Efficiency) (keeps zeros valid)
    df['LogEfficiency'] = np.log1p(df['Efficiency'].astype(float))

    # Create binary sex variable: Sex_Male = 1 if male, 0 if female
    # be robust to case variations and missing labels
    df['sex_str'] = df['sex'].astype(str).str.strip().str.lower()
    df['Sex_Male'] = df['sex_str'].map(lambda x: 1 if x.startswith('m') else (0 if x.startswith('f') else np.nan))

    # Map help to binary: Help_Received = 1 for yes-like answers, 0 otherwise
    df['help_str'] = df['help'].astype(str).str.strip().str.lower()
    df['Help_Received'] = df['help_str'].map(lambda x: 1 if x.startswith('y') else (0 if x.startswith('n') else np.nan))

    # Drop rows where mapping was ambiguous
    df = df.dropna(subset=['Sex_Male', 'Help_Received'])

    # Cast binary columns to integer dtype
    df['Sex_Male'] = df['Sex_Male'].astype(int)
    df['Help_Received'] = df['Help_Received'].astype(int)

    # Ensure chimpanzee is categorical (grouping variable for mixed model)
    df['chimpanzee'] = df['chimpanzee'].astype('category')

    # Ensure hammer is categorical
    df['hammer'] = df['hammer'].astype('category')

    # Keep only columns needed for modeling plus some diagnostics
    keep_cols = ['chimpanzee', 'age', 'Sex_Male', 'Help_Received', 'hammer', 'nuts_opened', 'seconds', 'Efficiency', 'LogEfficiency']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects model predicting LogEfficiency from age, sex, and help,
    controlling for hammer type and including a random intercept for each chimpanzee.

    Returns the fitted model object(s). Prints a brief summary.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Check required columns exist
    required = ['LogEfficiency', 'age', 'Sex_Male', 'Help_Received', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe missing columns required for modeling: {missing}")

    # Build formula: include hammer as categorical control via C(hammer)
    formula = 'LogEfficiency ~ age + Sex_Male + Help_Received + C(hammer)'

    # Fit mixed effects model with random intercepts for chimpanzee
    # Attempt several fitting strategies to avoid singular matrix / convergence failures
    md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])

    mdf = None
    fit_errors = []
    # Sequence of fit attempts with different options
    fit_attempts = [
        {'method': 'lbfgs', 'reml': False, 'options': {'maxiter': 1000}},
        {'method': 'lbfgs', 'reml': True, 'options': {'maxiter': 1000}},
        {'method': 'powell', 'reml': False, 'options': {'maxiter': 1000}},
        {'method': 'cg', 'reml': False, 'options': {'maxiter': 1000}},
        {'method': 'bfgs', 'reml': False, 'options': {'maxiter': 1000}},
        # final naive attempt: allow default settings
        {'method': None, 'reml': False, 'options': {}},
    ]

    for attempt in fit_attempts:
        try:
            if attempt['method'] is None:
                mdf = md.fit(reml=attempt['reml'])
            else:
                mdf = md.fit(method=attempt['method'], reml=attempt['reml'], **({'options': attempt['options']} if attempt.get('options') else {}))
            # If fit succeeded, break
            break
        except Exception as e:
            # Record error and try next strategy
            fit_errors.append((attempt, str(e)))
            mdf = None
            continue

    # Print concise summary if mixed model fitted successfully
    if mdf is not None:
        try:
            print(mdf.summary())
        except Exception:
            # If summary fails, just print that fit succeeded
            print("MixedLM fit succeeded; summary not available.")

    else:
        # If all mixed model attempts failed, report and proceed to OLS fallback
        print("MixedLM failed to converge or produced a singular matrix with attempted optimizers. Falling back to cluster-robust OLS.")
        for a, err in fit_errors:
            print(f"Attempt: {a} -> Error: {err}")

    # Also compute and print robust (clustered) standard errors by chimpanzee using OLS as a check
    ols_mod = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols_mod = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})
        print('\nCluster-robust OLS (clustered by chimpanzee) summary:')
        if ols_mod is not None:
            print(ols_mod.summary())
    except Exception as e:
        print(f"Cluster-robust OLS failed: {e}")
        ols_mod = None

    # Return both fitted mixed model (or None if failed) and cluster-robust OLS as a tuple for further inspection
    return {
        'mixedlm_result': mdf,
        'cluster_robust_ols': ols_mod
    }