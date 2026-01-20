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
    Transform the raw dataset into the final dataframe for modeling.

    Output columns used in the model:
      - age (kept as numeric)
      - sex_male (0/1)
      - HelpReceived (0/1)
      - Efficiency (nuts per second)
      - LogEfficiency (log1p(Efficiency)) -> DV
      - hammer (categorical control)
      - chimpanzee (grouping variable for random intercept)
    """
    df = df.copy()

    # Keep only rows with the fields necessary to compute efficiency and predictors
    required = ['nuts_opened', 'seconds', 'age', 'sex', 'help', 'hammer', 'chimpanzee']
    df = df.dropna(subset=required)

    # Ensure numeric types
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop any rows with missing / invalid numeric values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age'])

    # Remove sessions with non-positive duration
    df = df[df['seconds'] > 0]

    # Compute efficiency: nuts opened per second
    df['Efficiency'] = df['nuts_opened'] / df['seconds']

    # Log-transform for modeling (use log1p to handle zero efficiency safely)
    df['LogEfficiency'] = np.log1p(df['Efficiency'])

    # Binary encoding for sex: male = 1, female = 0
    # Handle possible whitespace / capitalization in the sex column
    df['sex_male'] = df['sex'].astype(str).str.strip().str.lower().map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # Binary encoding for help: y/yes -> 1, n/no -> 0
    df['HelpReceived'] = df['help'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})

    # Convert hammer to categorical (keeps original levels)
    df['hammer'] = df['hammer'].astype('category')

    # Ensure chimpanzee is treated as an identifier (string or category)
    df['chimpanzee'] = df['chimpanzee'].astype(str)

    # Drop rows where mapping produced NA for sex_male or HelpReceived
    df = df.dropna(subset=['sex_male', 'HelpReceived'])

    # Remove any infinite or NaN LogEfficiency rows (safety)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['LogEfficiency'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects model predicting LogEfficiency (DV) from age, sex, and help,
    controlling for hammer type (fixed effect) and including a random intercept for chimpanzee.

    Model: LogEfficiency ~ age + sex_male + HelpReceived + hammer  (random intercept by chimpanzee)

    Returns the fitted MixedLMResults object.
    """
    import statsmodels.formula.api as smf

    # Ensure the dataframe contains the columns we expect
    needed = ['LogEfficiency', 'age', 'sex_male', 'HelpReceived', 'hammer', 'chimpanzee']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Fit the mixed-effects model with a random intercept for each chimpanzee
    formula = 'LogEfficiency ~ age + sex_male + HelpReceived + hammer'

    # Use groups=df['chimpanzee'] to specify random intercept
    md = smf.mixedlm(formula, df, groups=df['chimpanzee'])
    try:
        mdf = md.fit(reml=False, method='lbfgs')
    except Exception:
        # Fallback to default fitting if lbfgs fails
        mdf = md.fit(reml=False)

    return mdf


