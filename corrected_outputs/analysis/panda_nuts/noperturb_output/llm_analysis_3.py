from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/noperturb_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Creates:
    - efficiency: nuts_opened / seconds
    - log_efficiency: log(efficiency + small_const)
    - HelpReceived: binary indicator (1 if help == 'y', 0 otherwise)

    Ensures categorical columns are in appropriate dtype and drops rows with missing or invalid critical values.
    """
    df = df.copy()

    # Drop rows missing critical measurement columns
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])

    # Ensure numeric columns are numeric
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows where conversion produced NaNs
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Remove sessions with non-positive duration (cannot compute rate)
    df = df[df['seconds'] > 0]

    # Compute raw efficiency (nuts per second) and log-transform it
    eps = 1e-6
    df['efficiency'] = df['nuts_opened'] / df['seconds']
    # Replace non-positive efficiencies (shouldn't occur if seconds>0 and nuts_opened>=0) with small positive constant
    df.loc[df['efficiency'] <= 0, 'efficiency'] = eps
    df['log_efficiency'] = np.log(df['efficiency'] + eps)

    # Standardize help coding into a binary column HelpReceived: 1 if help indicated (y), 0 otherwise
    # Lowercase to be robust to 'Y','y','N','n' etc.
    df['HelpReceived'] = df['help'].astype(str).str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    # For any unmapped values, default to 0 (no help) but keep a warning if needed
    df['HelpReceived'] = df['HelpReceived'].fillna(0).astype(int)

    # Ensure sex is a clean categorical with lowercase 'f' or 'm'
    df['sex'] = df['sex'].astype(str).str.lower().str.strip()

    # Ensure hammer is a string categorical
    df['hammer'] = df['hammer'].astype(str).str.strip()

    # Ensure chimpanzee id is an integer (grouping variable)
    # If chimpanzee ids are non-numeric strings, keep as-is (MixedLM accepts group labels), but try to coerce to int when possible
    try:
        df['chimpanzee'] = pd.to_numeric(df['chimpanzee'], errors='coerce').astype('Int64')
        # if coercion produced NaN for some rows, fill those with original string ids
        mask_non_numeric = df['chimpanzee'].isna()
        if mask_non_numeric.any():
            # fall back to original string values for those rows
            orig_ids = df.loc[mask_non_numeric, 'chimpanzee'].astype(str)
            df.loc[mask_non_numeric, 'chimpanzee'] = orig_ids
        # finally, convert to plain object dtype for grouping
        df['chimpanzee'] = df['chimpanzee'].astype(object)
    except Exception:
        df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Final sanity drop: ensure we still have required columns
    required = ['log_efficiency', 'age', 'sex', 'HelpReceived', 'hammer', 'chimpanzee']
    df = df.dropna(subset=['log_efficiency', 'age', 'sex', 'HelpReceived', 'hammer', 'chimpanzee'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects model predicting log_efficiency from age, sex, and help
    while controlling for hammer type and including a random intercept for chimpanzee.

    Returns the fitted model results object. If there are fewer than 2 groups for random
    effects, falls back to an OLS model with clustered standard errors by chimpanzee.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure input appears transformed (quick checks)
    required_cols = ['log_efficiency', 'age', 'sex', 'HelpReceived', 'hammer', 'chimpanzee']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Convert categorical predictors to categorical dtype for formula handling
    df = df.copy()
    df['sex'] = df['sex'].astype('category')
    df['hammer'] = df['hammer'].astype('category')
    # Ensure HelpReceived is numeric
    df['HelpReceived'] = pd.to_numeric(df['HelpReceived'], errors='coerce').fillna(0).astype(int)

    # Decide between mixed effects (random intercept by chimpanzee) and OLS fallback
    n_groups = df['chimpanzee'].nunique()
    formula = 'log_efficiency ~ age + C(sex) + HelpReceived + C(hammer)'

    if n_groups >= 2:
        # Mixed effects with random intercept by chimpanzee
        md = sm.MixedLM.from_formula(formula, groups='chimpanzee', data=df)
        mdf = md.fit(reml=False)
        print(mdf.summary())
        return mdf
    else:
        # Fallback: OLS. Use cluster-robust SE by chimpanzee if possible.
        ols_mod = smf.ols(formula, data=df).fit()
        try:
            clustered = ols_mod.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
            print(clustered.summary())
            return clustered
        except Exception:
            print(ols_mod.summary())
            return ols_mod


