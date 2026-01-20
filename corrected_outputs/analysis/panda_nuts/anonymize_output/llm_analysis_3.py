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
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/anonymize_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.

    Produces these columns used in modeling:
      - ID: subject identifier (from feature1)
      - Age: numeric age in years (from feature2)
      - Age_c: age centered around the sample mean
      - Sex_male: binary indicator (1 = male, 0 = female) from feature3
      - HammerType: hammer type (categorical) from feature4
      - NutsOpened: number of nuts opened in session (from feature5)
      - DurationSec: session duration in seconds (from feature6)
      - Help: binary indicator (1 = received help, 0 = no help) from feature7
      - Efficiency_NutsPerMin: NutsOpened / (DurationSec/60)

    Rows with missing or invalid critical fields are dropped.
    """
    df = df.copy()

    # Rename raw columns to analysis-friendly names
    rename_map = {
        'feature1': 'ID',
        'feature2': 'Age',
        'feature3': 'Sex',
        'feature4': 'HammerType',
        'feature5': 'NutsOpened',
        'feature6': 'DurationSec',
        'feature7': 'HelpRaw'
    }
    df = df.rename(columns=rename_map)

    # Drop rows missing essential columns
    df = df.dropna(subset=['ID', 'Age', 'Sex', 'NutsOpened', 'DurationSec', 'HelpRaw'])

    # Coerce numeric types
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['NutsOpened'] = pd.to_numeric(df['NutsOpened'], errors='coerce')
    df['DurationSec'] = pd.to_numeric(df['DurationSec'], errors='coerce')

    # Normalize Sex to binary: male=1, female=0
    df['Sex_male'] = df['Sex'].astype(str).str.strip().str.lower().map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # Normalize Help indicator to binary: y/yes -> 1, n/no -> 0
    df['Help'] = df['HelpRaw'].astype(str).str.strip().str.lower().map({
        'y': 1, 'yes': 1, 't': 1, 'true': 1,
        'n': 0, 'no': 0, 'f': 0, 'false': 0
    })

    # Remove non-positive durations (can't compute rate) by setting to NaN
    df.loc[df['DurationSec'] <= 0, 'DurationSec'] = np.nan

    # Efficiency: nuts opened per minute
    df['Efficiency_NutsPerMin'] = df['NutsOpened'] / (df['DurationSec'] / 60.0)

    # Drop any rows with missing derived or essential variables
    df = df.dropna(subset=['Efficiency_NutsPerMin', 'Sex_male', 'Help', 'Age'])

    # Center age for interpretability in interaction terms
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Ensure HammerType is string/categorical
    df['HammerType'] = df['HammerType'].astype(str)

    # Keep only relevant columns for modeling (plus HelpRaw if user wants raw values)
    out_cols = ['ID', 'Age', 'Age_c', 'Sex_male', 'HammerType', 'NutsOpened', 'DurationSec', 'Help', 'HelpRaw', 'Efficiency_NutsPerMin']
    # Some source datasets might not have HelpRaw after earlier operations; guard against that
    out_cols = [c for c in out_cols if c in df.columns]

    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a model to estimate effects of age, sex, and receiving help on nut-cracking efficiency.

    Strategy:
      - If there are repeated observations per individual (ID), fit a linear mixed effects model
        with a random intercept for ID to account for within-individual correlation.
      - If the mixed model fails to converge or produces a singular Hessian, fall back to OLS.

    Model formula includes interactions to test whether the effect of Help depends on Age or Sex:
      Efficiency_NutsPerMin ~ Age_c + Sex_male + Help + Age_c:Help + Sex_male:Help + C(HammerType)

    Returns the fitted results object (MixedLMResults or RegressionResults).
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Ensure ID exists
    if 'ID' not in df.columns:
        raise ValueError("Input dataframe must contain 'ID' column (see transform output).")

    formula = 'Efficiency_NutsPerMin ~ Age_c + Sex_male + Help + Age_c:Help + Sex_male:Help + C(HammerType)'

    n_obs = len(df)
    n_ids = int(df['ID'].nunique())

    # If there is more than one observation per ID, try mixed effects to account for repeated measures
    if n_ids < n_obs:
        md = smf.mixedlm(formula, df, groups=df['ID'])
        # Try a sequence of fitting strategies; if all fail or lead to singular Hessian, fall back to OLS
        fit_exceptions = []
        try_methods = [
            {'reml': False, 'method': 'lbfgs', 'maxiter': 200},
            {'reml': False, 'method': 'powell', 'maxiter': 2000},
            {'reml': True, 'method': 'lbfgs', 'maxiter': 200}
        ]
        for opts in try_methods:
            try:
                mdf = md.fit(**opts)
                return mdf
            except np.linalg.LinAlgError as e:
                fit_exceptions.append(e)
                warnings.warn(f"MixedLM linear algebra error with options {opts}: {e}. Trying fallback.")
            except Exception as e:
                fit_exceptions.append(e)
                warnings.warn(f"MixedLM fitting failed with options {opts}: {e}. Trying next option.")
        # If we reach here, mixed model attempts failed; fall back to OLS
        warnings.warn("Falling back to OLS because MixedLM failed to fit reliably.")
        ols_res = smf.ols(formula, data=df).fit()
        return ols_res
    else:
        # No repeated measures: ordinary least squares
        ols_res = smf.ols(formula, data=df).fit()
        return ols_res