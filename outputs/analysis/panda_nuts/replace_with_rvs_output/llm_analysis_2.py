from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/replace_with_rvs_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.

    Produces the following additional columns used in the model:
      - efficiency: nuts_opened / seconds (raw rate)
      - log_efficiency: log((nuts_opened + 0.5) / seconds)
      - age_c: age centered around the sample mean
      - sex_m: binary indicator (male=1, female=0)
      - help_y: binary indicator (help received=1, no help=0)
      - hammer: categorical (unchanged type but ensured dtype)
      - chimpanzee: categorical id (ensured dtype)
    Rows with missing critical fields or non-positive session durations are removed.
    """
    df = df.copy()

    # Drop rows with missing essential data
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'chimpanzee'])

    # Remove invalid session durations (avoid division by zero)
    df = df[df['seconds'] > 0]

    # Raw efficiency (nuts per second)
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Log-transform efficiency with a small offset to handle zero counts
    df['log_efficiency'] = np.log((df['nuts_opened'] + 0.5) / df['seconds'])

    # Encode sex: male = 1, female = 0
    # Normalize common variations and map; unknown values become NaN and will be dropped above
    df['sex_m'] = df['sex'].astype(str).str.strip().str.lower().map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # Encode help: 'y' or 'Y' (yes) -> 1, 'n' or 'N' (no) -> 0
    df['help_y'] = df['help'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})

    # Ensure hammer is categorical (keeps original levels)
    df['hammer'] = df['hammer'].astype('category')

    # Center age (improves interpretability of intercept)
    df['age_c'] = df['age'] - df['age'].mean()

    # Ensure chimpanzee id is categorical for grouping
    df['chimpanzee'] = df['chimpanzee'].astype('category')

    # After derived encodings, drop any rows where mappings failed (NaNs in encoded variables)
    df = df.dropna(subset=['sex_m', 'help_y'])

    # Keep only columns necessary for modeling plus a few useful originals
    keep_cols = ['chimpanzee', 'age', 'age_c', 'sex', 'sex_m', 'help', 'help_y', 'hammer', 'nuts_opened', 'seconds', 'efficiency', 'log_efficiency']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting log_efficiency from age, sex, help and hammer type
    with a random intercept for each chimpanzee.

    Model specification (fixed effects):
      log_efficiency ~ age_c + sex_m + help_y + C(hammer)
    Random effects: random intercept for chimpanzee (groups=chimpanzee)

    Returns the fitted MixedLMResults object (from statsmodels) so the caller can inspect summary, params, etc.
    """
    import statsmodels.formula.api as smf

    # Verify required columns exist
    required = ['log_efficiency', 'age_c', 'sex_m', 'help_y', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit linear mixed-effects model with chimpanzee random intercept
    # Use REML=False to make results comparable to ML for fixed effects hypothesis testing
    formula = 'log_efficiency ~ age_c + sex_m + help_y + C(hammer)'
    md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
    mdf = md.fit(reml=False)

    # Return the fitted model object; caller can call mdf.summary() or inspect coefficients
    return mdf


