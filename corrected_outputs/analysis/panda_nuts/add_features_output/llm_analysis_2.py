from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/add_features_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a cleaned dataframe with columns required for modeling.
    Outputs the following columns (used in the model):
      - chimpanzee: categorical ID (grouping)
      - age: original age (kept for reference)
      - age_c: mean-centered age (used as IV)
      - sex: categorical sex (used as IV)
      - hammer: categorical hammer type (control)
      - nuts_opened: numeric (raw)
      - seconds: numeric session duration
      - nuts_per_sec: nuts_opened / seconds (efficiency rate)
      - log_nuts_per_sec: np.log1p(nuts_per_sec) (DV)
      - Help: binary indicator (1 if help == 'y', 0 if help == 'n') (IV)
    """
    df = df.copy()

    # Keep only rows with required columns present
    required_cols = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    df = df.dropna(subset=required_cols)

    # Ensure numeric columns are numeric
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Remove sessions with non-positive duration
    df = df[df['seconds'] > 0].copy()

    # Standardize and clean categorical text fields
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    df['hammer'] = df['hammer'].astype(str).str.strip()
    df['help'] = df['help'].astype(str).str.strip()

    # Map help to binary indicator. Accept common variants; drop rows that cannot be mapped.
    help_map = {'y': 1, 'yes': 1, 'n': 0, 'no': 0}
    df['Help'] = df['help'].str.lower().map(help_map)
    df = df.dropna(subset=['Help'])

    # Compute efficiency (nuts per second) and log-transform the rate to stabilize variance
    df['nuts_per_sec'] = df['nuts_opened'] / df['seconds']
    df['log_nuts_per_sec'] = np.log1p(df['nuts_per_sec'])

    # Center age for interpretability
    df['age_c'] = df['age'] - df['age'].mean()

    # Convert grouping/categorical vars to appropriate dtypes for modeling
    df['chimpanzee'] = df['chimpanzee'].astype('category')
    df['sex'] = df['sex'].astype('category')
    df['hammer'] = df['hammer'].astype('category')

    # Select and return only columns needed for downstream analysis
    out_cols = ['chimpanzee', 'age', 'age_c', 'sex', 'hammer', 'nuts_opened', 'seconds', 'nuts_per_sec', 'log_nuts_per_sec', 'Help']
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting log_nuts_per_sec from age (centered), sex, Help, and hammer,
    with a random intercept for chimpanzee to account for repeated measures.

    Model:
      log_nuts_per_sec ~ age_c + sex + Help + hammer + (1 | chimpanzee)

    Returns the fitted MixedLMResults object.
    """
    import statsmodels.formula.api as smf

    # Copy dataframe to avoid side effects
    df = df.copy()

    # Ensure the model columns exist
    required = ['log_nuts_per_sec', 'age_c', 'sex', 'Help', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit mixed effects model with random intercept for chimpanzee
    # Treat sex and hammer as categorical factors (patsy will handle categorical dtype)
    md = smf.mixedlm("log_nuts_per_sec ~ age_c + sex + Help + hammer", data=df, groups=df["chimpanzee"])
    mdf = md.fit(reml=False)

    # Print and return results
    print(mdf.summary())
    return mdf


