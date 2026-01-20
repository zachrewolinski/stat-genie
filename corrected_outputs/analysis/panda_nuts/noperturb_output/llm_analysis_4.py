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
    Transform the raw dataframe for modelling.

    Produces the following columns used in the model:
      - NutsPerSec : nuts_opened / seconds (for inspection/plotting)
      - LogEff     : log(nuts_opened + 1) - log(seconds) (dependent variable)
      - Sex_M      : binary sex indicator (1 = male, 0 = female)
      - Help_Y     : binary help indicator (1 = received help, 0 = no help)
      - hammer     : ensured as string/categorical (control)
      - chimpanzee : ensured as integer/group id (random effect)

    Drops rows with missing essential data and sessions with non-positive seconds.
    """
    df = df.copy()

    # Drop rows missing essential variables
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Remove sessions with non-positive duration to avoid division issues
    df = df[df['seconds'] > 0].copy()

    # Derived measures
    # Rate (nuts per second) kept for inspection
    df['NutsPerSec'] = df['nuts_opened'] / df['seconds']

    # Log-transformed efficiency (handles zeros in nuts_opened): log(nuts_opened + 1) - log(seconds)
    df['LogEff'] = np.log(df['nuts_opened'] + 1.0) - np.log(df['seconds'])

    # Binary sex indicator: 1 = male, 0 = female. Map robustly by lowercasing string values.
    df['Sex_M'] = df['sex'].astype(str).str.lower().map({'m': 1, 'f': 0})
    # If mapping produced NaN (unexpected category), fill with 0 and warn implicitly
    df['Sex_M'] = df['Sex_M'].fillna(0).astype(int)

    # Binary help indicator: 1 = yes (y), 0 = no (n). Handle case-insensitive entries.
    df['Help_Y'] = df['help'].astype(str).str.lower().map({'y': 1, 'n': 0})
    df['Help_Y'] = df['Help_Y'].fillna(0).astype(int)

    # Ensure hammer is a string/categorical column (control). Keep original values (will be used with C(hammer) in model).
    df['hammer'] = df['hammer'].astype(str)

    # Ensure chimpanzee id is an integer (grouping variable for mixed model)
    df['chimpanzee'] = df['chimpanzee'].astype(int)

    # Return only rows and columns necessary for modeling (but keep original for traceability)
    # It's fine to return the full df copy; model function will select the columns it needs.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a mixed-effects linear model predicting log efficiency (LogEff).

    Model specification:
      LogEff ~ age + Sex_M + Help_Y + C(hammer) + (1 | chimpanzee)

    - Fixed effects: age (continuous), Sex_M (binary), Help_Y (binary), hammer (categorical)
    - Random effects: random intercept for chimpanzee to account for repeated measures

    Returns the fitted model results object. If the mixed model fails to converge, falls back to an OLS with cluster-robust SEs by chimpanzee.
    """
    import statsmodels.formula.api as smf

    # Make a local copy
    df = df.copy()

    # Ensure that the dependent variable and predictors exist
    required = ['LogEff', 'age', 'Sex_M', 'Help_Y', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    formula = 'LogEff ~ age + Sex_M + Help_Y + C(hammer)'

    # Try fitting a linear mixed-effects model with a random intercept for chimpanzee
    try:
        md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
        mdf = md.fit(reml=False)
        return mdf
    except Exception as e:
        # Fallback: ordinary least squares with cluster-robust standard errors by chimpanzee
        # This will still provide inference that accounts for within-individual correlation.
        ols_mod = smf.ols(formula, data=df).fit()
        # Attach cluster-robust covariance to the result object for external inspection
        try:
            cov = ols_mod.get_robustcov_results(cov_type='cluster', groups=df['chimpanzee'])
            return cov
        except Exception:
            # If robust covariance fails, return the plain OLS fit
            return ols_mod


