from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/replace_and_positive_statement_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.
    Produces:
      - nuts_per_min: nuts opened per minute (continuous rate)
      - log_nuts_per_min: log-transformed rate for modeling
      - help_bin: binary indicator for receiving help (1 = yes, 0 = no)
      - age_c: centered age
      - sex, hammer, chimpanzee preserved/converted to categorical strings
    Drops rows with missing essential values (nuts_opened, seconds, age, sex, help).
    """
    df = df.copy()

    # Drop rows missing essential vars
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])

    # Ensure numeric for calculations
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df = df.dropna(subset=['nuts_opened', 'seconds'])

    # Compute rate: nuts per minute
    # avoid division by zero if any 'seconds' == 0 (shouldn't happen given schema, but guard anyway)
    df = df[df['seconds'] > 0]
    df['nuts_per_min'] = df['nuts_opened'] / df['seconds'] * 60.0

    # Log-transform the rate for modeling (small epsilon to avoid log(0))
    eps = 1e-6
    df['log_nuts_per_min'] = np.log(df['nuts_per_min'] + eps)

    # Create binary help indicator
    # Map common variants to 1/0; default to 0 for unknown/missing after mapping
    df['help_bin'] = df['help'].astype(str).str.strip().str.lower().map({
        'y': 1, 'yes': 1, 'yep': 1, 'true': 1, 't': 1, '1': 1,
        'n': 0, 'no': 0, 'false': 0, 'f': 0, '0': 0
    })
    df['help_bin'] = df['help_bin'].fillna(0).astype(int)

    # Standardize categorical columns: keep as strings for formula handling
    df['sex'] = df['sex'].astype(str).str.strip()
    df['hammer'] = df['hammer'].astype(str).str.strip()
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Center age for interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Final: drop any rows with NaN in new columns (defensive)
    df = df.dropna(subset=['nuts_per_min', 'log_nuts_per_min', 'age_c', 'sex', 'hammer', 'chimpanzee'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting log(nuts_per_min) from age, sex, and help,
    controlling for hammer type and with a random intercept for chimpanzee.

    Model formula:
      log_nuts_per_min ~ age_c + C(sex) + help_bin + C(hammer)
    Random effects:
      random intercept for chimpanzee (groups=chimpanzee)

    Returns the fitted model results object (statsmodels MixedLMResults).
    Prints the model summary.
    """
    import statsmodels.formula.api as smf

    # Fit mixed-effects linear model (random intercept by chimpanzee)
    # Use ML (reml=False) for easier comparison across fixed-effects specifications
    md = smf.mixedlm("log_nuts_per_min ~ age_c + C(sex) + help_bin + C(hammer)", data=df, groups=df["chimpanzee"])

    try:
        mdf = md.fit(reml=False)
    except Exception as e:
        # If convergence issues occur, try a more robust optimizer
        mdf = md.fit(reml=False, method='nm', maxiter=2000)

    # Print summary for inspection
    print(mdf.summary())

    # For asserting effects of interest, we return the full results object
    return mdf


