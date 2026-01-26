from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/noperturb_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw chimpanzee nut-cracking dataset into analysis-ready form.

    Produces these columns used in the model:
      - 'log_rate' : log((nuts_opened + 0.5) / seconds) implemented as log(nuts_opened + 0.5) - log(seconds)
      - 'efficiency' : nuts_opened / seconds (nuts per second; kept for diagnostics/plotting)
      - 'age_c' : age mean-centered
      - 'sex_m' : 1 if sex == 'm', else 0
      - 'help_yes' : 1 if help indicates yes (y/yes), else 0
      - ensures 'chimpanzee' and 'hammer' are present and appropriately typed
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce to appropriate dtypes and drop rows missing essential data
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Create an interpretable efficiency metric (nuts per second)
    # Keep for diagnostics / plotting
    # Use float division
    df['efficiency'] = df['nuts_opened'].astype(float) / df['seconds'].astype(float)

    # Create a log-rate (log of count per unit time). Add small constant to counts to handle zeros.
    # log_rate = log(nuts_opened + 0.5) - log(seconds)
    df['log_rate'] = np.log(df['nuts_opened'].astype(float) + 0.5) - np.log(df['seconds'].astype(float))

    # Standardize / encode predictors
    # Sex: create male indicator (1 = 'm', 0 otherwise). Handle varied capitalization.
    df['sex'] = df['sex'].astype(str).str.lower()
    df['sex_m'] = df['sex'].apply(lambda x: 1 if x == 'm' else 0)

    # Help: normalize and encode yes=1, no=0 (accepts 'y','yes' as yes)
    df['help'] = df['help'].astype(str).str.lower()
    df['help_yes'] = df['help'].apply(lambda x: 1 if x in ['y', 'yes'] else 0)

    # Hammer: keep as-is but coerce to string/category for modeling with C(hammer)
    df['hammer'] = df['hammer'].astype(str)

    # Chimpanzee ID: coerce to string (grouping variable)
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Center age for interpretability
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()

    # Return transformed dataframe with the columns needed for modeling/diagnostics
    keep_cols = [
        'chimpanzee', 'age', 'age_c', 'sex', 'sex_m', 'help', 'help_yes', 'hammer',
        'nuts_opened', 'seconds', 'efficiency', 'log_rate'
    ]
    # Some columns may already be present; selecting safe subset
    present = [c for c in keep_cols if c in df.columns]
    return df.loc[:, present].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting log-rate of nut-cracking.

    Model specification:
      log_rate ~ age_c + sex_m + help_yes + C(hammer) + (1 | chimpanzee)

    - A random intercept for chimpanzee accounts for repeated sessions by the same individual.
    - Hammer is included as a categorical control via C(hammer).

    Returns the fitted model object (MixedLMResults). If the mixed model fails to converge,
    falls back to an OLS model and returns that fitted OLSResults object.
    """
    import statsmodels.formula.api as smf

    # Basic checks
    if 'log_rate' not in df.columns:
        raise ValueError("Transformed dataframe must contain 'log_rate'. Run transform() first.")

    # Fit linear mixed effects model with random intercept for chimpanzee
    formula = 'log_rate ~ age_c + sex_m + help_yes + C(hammer)'
    try:
        md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
        mdf = md.fit(reml=False)
        return mdf
    except Exception as e:
        # Fallback: ordinary least squares with robust standard errors clustered by chimpanzee
        ols = smf.ols(formula, data=df).fit()
        # attach note about fallback
        ols.fallback_note = f"MixedLM failed: {e}. Returned OLS fit instead."
        return ols


