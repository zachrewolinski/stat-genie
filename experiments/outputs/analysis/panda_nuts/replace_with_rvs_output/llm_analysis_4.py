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
    Transform raw nut-cracking dataset into analysis-ready dataframe.

    Steps performed:
    - Copy input to avoid side effects
    - Drop rows missing essential columns
    - Ensure numeric columns are numeric and seconds > 0
    - Create binary indicators: sex_male, help_received
    - Compute rate_per_sec = (nuts_opened + 0.5) / seconds and log_rate = log(rate_per_sec)
    - Ensure hammer and chimpanzee are treated as categorical-like columns

    Final dataframe contains the columns used in the model: ['chimpanzee','age','sex_male','help_received','hammer','nuts_opened','seconds','rate_per_sec','log_rate']
    """
    df = df.copy()

    # Required columns
    required_cols = ['chimpanzee', 'age', 'sex', 'help', 'nuts_opened', 'seconds', 'hammer']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing essential data
    df = df.dropna(subset=['chimpanzee', 'age', 'sex', 'help', 'nuts_opened', 'seconds'])

    # Convert to numeric where appropriate
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Remove sessions with non-positive duration
    df = df[df['seconds'] > 0]

    # Create binary sex indicator: sex_male = 1 if male, 0 if female
    df['sex_male'] = (
        df['sex'].astype(str).str.strip().str.lower().map({'m': 1, 'male': 1, 'f': 0, 'female': 0})
    )
    # If mapping produced NaN (unexpected labels), try a conservative fallback by treating unknown as NA
    if df['sex_male'].isnull().any():
        df = df.dropna(subset=['sex_male'])
    df['sex_male'] = df['sex_male'].astype(int)

    # Create binary help indicator: help_received = 1 if 'y'/'yes', 0 if 'n'/'no'
    df['help_received'] = (
        df['help'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    )
    # Drop rows with ambiguous help coding
    if df['help_received'].isnull().any():
        df = df.dropna(subset=['help_received'])
    df['help_received'] = df['help_received'].astype(int)

    # Ensure hammer is a string/categorical
    df['hammer'] = df['hammer'].astype(str)

    # Ensure chimpanzee ID is categorical (keeps original IDs but as category)
    df['chimpanzee'] = df['chimpanzee'].astype('category')

    # Compute rate (per second) and log-rate. Add 0.5 to nuts_opened to stabilize zeros.
    df['rate_per_sec'] = (df['nuts_opened'] + 0.5) / df['seconds']
    # Guard against non-positive rate (shouldn't happen because seconds > 0 and numerator >= 0.5)
    df['log_rate'] = np.log(df['rate_per_sec'])

    # Return only the columns needed for the model (keeps extras useful for diagnostics)
    final_cols = ['chimpanzee', 'age', 'sex_male', 'help_received', 'hammer', 'nuts_opened', 'seconds', 'rate_per_sec', 'log_rate']
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting log_rate from age, sex, and help (with hammer as a control),
    including a random intercept for chimpanzee to account for repeated observations.

    Model formula:
      log_rate ~ age + sex_male + help_received + C(hammer)
    Random effects:
      (1 | chimpanzee)

    Returns:
      The fitted MixedLMResults object (mdf).
    """
    import statsmodels.formula.api as smf

    # Check that expected columns exist
    expected = ['log_rate', 'age', 'sex_male', 'help_received', 'hammer', 'chimpanzee']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Fit the mixed effects model. Use REML=False (ML) so comparisons with nested models are straightforward.
    # C(hammer) treats hammer as categorical control.
    md = smf.mixedlm("log_rate ~ age + sex_male + help_received + C(hammer)", data=df, groups=df["chimpanzee"]) 
    mdf = md.fit(reml=False)

    # Print and return the fitted model results
    print(mdf.summary())
    return mdf


