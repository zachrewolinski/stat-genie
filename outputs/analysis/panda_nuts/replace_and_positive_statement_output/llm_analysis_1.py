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
    Transform the raw dataset into the analysis dataframe. Creates efficiency measures, encodes categorical predictors,
    and drops rows with missing/invalid values required for the model.

    Required output columns (used in the model):
      - NutsPerSec: nuts_opened / seconds
      - LogNutsPerSec: natural log of NutsPerSec (with small epsilon added)
      - age: original age column (kept)
      - Sex_m: indicator (1=male, 0=female)
      - Help_y: indicator (1=help received 'y', 0=otherwise)
      - hammer: hammer type (string/categorical)
      - chimpanzee: ID for random effects grouping
    """
    df = df.copy()

    # Standardize string columns and handle common variants
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    if 'help' in df.columns:
        df['help'] = df['help'].astype(str).str.strip().str.lower()
    if 'hammer' in df.columns:
        df['hammer'] = df['hammer'].astype(str).str.strip()

    # Drop rows missing the primary fields required to compute efficiency or predictors
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee'])

    # Remove zero or negative session durations to avoid division by zero
    df = df[df['seconds'] > 0]

    # Compute nuts opened per second as primary efficiency measure
    df['NutsPerSec'] = df['nuts_opened'] / df['seconds']

    # Log-transform the rate to stabilize variance; add small epsilon to avoid log(0)
    eps = 1e-6
    df['LogNutsPerSec'] = np.log(df['NutsPerSec'] + eps)

    # Encode binary predictors as numeric indicators for modeling
    # Sex: 'm' -> 1, others (including 'f') -> 0
    df['Sex_m'] = df['sex'].map({'m': 1, 'f': 0})
    # If mapping produced NaN (unexpected values), coerce these to 0 and keep a warning
    df['Sex_m'] = df['Sex_m'].fillna(0).astype(int)

    # Help: 'y' -> 1, anything else -> 0 (handles uppercase/lowercase earlier)
    df['Help_y'] = df['help'].map({'y': 1})
    df['Help_y'] = df['Help_y'].fillna(0).astype(int)

    # Ensure hammer is a string categorical column for model formula use
    df['hammer'] = df['hammer'].astype(str)

    # Ensure chimpanzee ID is present and of appropriate dtype
    # (Mixed models accept numeric or string group labels)
    df['chimpanzee'] = df['chimpanzee']

    # Final drop of any rows that still contain NA in model columns
    model_cols = ['LogNutsPerSec', 'age', 'Sex_m', 'Help_y', 'hammer', 'chimpanzee']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a mixed-effects model to test whether age, sex, and receiving help predict nut-cracking efficiency.

    Model specification:
      - Dependent variable: LogNutsPerSec (log of nuts per second)
      - Fixed effects: age (continuous), Sex_m (0/1), Help_y (0/1), C(hammer) (categorical control)
      - Random effects: random intercept for chimpanzee to account for repeated measures

    Returns the fitted mixed model results object (statsmodels MixedLMResults).
    """
    import statsmodels.formula.api as smf

    # Copy to avoid side effects
    df = df.copy()

    # Ensure required columns exist
    required = ['LogNutsPerSec', 'age', 'Sex_m', 'Help_y', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Specify formula: include hammer as a categorical control
    formula = 'LogNutsPerSec ~ age + Sex_m + Help_y + C(hammer)'

    # Fit mixed-effects model with random intercept for chimpanzee
    # Use reml=False for likelihood-based comparisons if desired
    md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
    mdf = md.fit(reml=False)

    # Print brief summary to console and return the fitted object for programmatic inspection
    try:
        print(mdf.summary())
    except Exception:
        # summary printing can sometimes fail in limited environments; ignore
        pass

    return mdf


