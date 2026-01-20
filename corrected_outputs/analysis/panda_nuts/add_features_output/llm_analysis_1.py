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
    Transform the raw dataset into the analysis dataframe.

    Output columns used in the model:
      - chimpanzee: group id for mixed model (unchanged except type cast)
      - age: numeric age in years
      - hammer: categorical hammer type (kept as-is for C(hammer) in formula)
      - nuts_opened: numeric count of nuts opened
      - seconds: numeric session duration in seconds
      - NutsPerSec: nuts_opened / seconds (raw rate)
      - LogNutsPerSec: log((nuts_opened + 0.5) / seconds) -- dependent variable
      - sex_M: 1 if male, 0 if female
      - help_Y: 1 if received help (y/yes), 0 otherwise
    """
    # make a copy to avoid modifying the original
    df = df.copy()

    # Keep columns required for the analysis; drop rows with missing critical fields
    required = ['chimpanzee', 'age', 'sex', 'hammer', 'nuts_opened', 'seconds', 'help']
    df = df.dropna(subset=required)

    # Ensure numeric columns are numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['age', 'nuts_opened', 'seconds'])

    # Remove sessions with non-positive duration
    df = df[df['seconds'] > 0]

    # Compute per-second rate and log-transformed rate (DV)
    # Add a small offset to counts (0.5) to stabilize zeros in log transform
    df['NutsPerSec'] = df['nuts_opened'] / df['seconds']
    df['LogNutsPerSec'] = np.log((df['nuts_opened'] + 0.5) / df['seconds'])

    # Clean and encode sex -> sex_M (1 = male, 0 = female)
    df['sex_clean'] = df['sex'].astype(str).str.strip().str.lower()
    df['sex_M'] = df['sex_clean'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # Clean and encode help -> help_Y (1 = yes, 0 = no)
    df['help_clean'] = df['help'].astype(str).str.strip().str.lower()
    df['help_Y'] = df['help_clean'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})

    # Drop rows where mapping failed (unexpected category values)
    df = df.dropna(subset=['sex_M', 'help_Y'])

    # Convert to integer type
    df['sex_M'] = df['sex_M'].astype(int)
    df['help_Y'] = df['help_Y'].astype(int)

    # Ensure chimpanzee is treated as a categorical/grouping variable (keep original values but ensure type)
    # Many modeling functions accept integer or string group labels; keep as-is but cast to string for safety
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Keep only the columns necessary for modeling (but retain a few raw columns for diagnostics if needed)
    keep = ['chimpanzee', 'age', 'sex_M', 'help_Y', 'hammer', 'nuts_opened', 'seconds', 'NutsPerSec', 'LogNutsPerSec']
    df = df[keep]

    # Reset index before returning
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear mixed-effects model predicting log rate of nuts opened per second
    from age, sex, and help, controlling for hammer type and with a random intercept
    for chimpanzee to account for repeated observations.

    Model formula:
      LogNutsPerSec ~ age + sex_M + help_Y + C(hammer)
    Random effects:
      random intercept for chimpanzee

    Returns the fitted model results object (statsmodels MixedLMResults).
    """
    import statsmodels.formula.api as smf

    # Ensure the required columns are present
    required_cols = ['LogNutsPerSec', 'age', 'sex_M', 'help_Y', 'hammer', 'chimpanzee']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"The dataframe is missing required columns for modeling: {missing}")

    # Fit the mixed-effects model (use REML=False for likelihood-based inference; method='lbfgs' for stability)
    formula = 'LogNutsPerSec ~ age + sex_M + help_Y + C(hammer)'
    md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
    try:
        mdf = md.fit(reml=False, method='lbfgs')
    except Exception:
        # Fallback to default fit in case of optimizer issues
        mdf = md.fit(reml=False)

    # Return the fitted model results object; the caller can inspect mdf.summary()
    return mdf


