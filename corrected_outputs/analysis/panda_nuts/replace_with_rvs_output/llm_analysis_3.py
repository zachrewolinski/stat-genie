from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/replace_with_rvs_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw nut-cracking dataset into a dataframe ready for analysis.

    Creates:
    - Efficiency: nuts_opened / seconds (nuts per second)
    - Sex_M: binary 1 if male ('m'), 0 if female ('f')
    - HelpReceived: binary 1 if help == 'y' (or 'Y'), 0 if help == 'n' (or 'N')
    - Hammer_G, Hammer_Q, Hammer_wood: binary dummies for common hammer types (baseline = other types)

    Filters out sessions with nonpositive or missing seconds and rows missing nuts_opened.
    """
    # Make a copy to avoid mutating original
    df = df.copy()

    # Ensure numeric columns are numeric
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['seconds'] = pd.to_numeric(df['seconds'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Drop rows that cannot be used to compute efficiency
    df = df.dropna(subset=['nuts_opened', 'seconds'])
    # Remove sessions with zero or negative duration
    df = df[df['seconds'] > 0]

    # Efficiency: nuts opened per second
    df['Efficiency'] = df['nuts_opened'] / df['seconds']

    # Sex binary: 1 = male, 0 = female. Coerce to lowercase then map.
    df['Sex_M'] = df['sex'].astype(str).str.lower().map({'m': 1, 'f': 0})

    # Help binary: 1 = received help ('y' or 'Y'), 0 = no ('n' or 'N')
    df['HelpReceived'] = df['help'].astype(str).str.lower().map({'y': 1, 'n': 0})

    # Create explicit hammer-type dummy columns for common types observed in data
    # We set 1 for the indicated type, 0 otherwise. Any hammer type not in these three
    # will be treated as the omitted baseline.
    df['Hammer_G'] = (df['hammer'].astype(str) == 'G').astype(int)
    df['Hammer_Q'] = (df['hammer'].astype(str) == 'Q').astype(int)
    # Some rows use 'wood' (lowercase) in the sample; match case-insensitively
    df['Hammer_wood'] = df['hammer'].astype(str).str.lower().eq('wood').astype(int)

    # It's useful to drop rows missing critical predictors (age, Sex_M, HelpReceived)
    # for the model. We will drop rows where any of the model's predictors are missing.
    required_cols = ['age', 'Sex_M', 'HelpReceived', 'Efficiency', 'chimpanzee']
    df = df.dropna(subset=required_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting Efficiency (nuts/sec) from age, sex, and help,
    controlling for hammer type and including a random intercept for chimpanzee.

    Model (in words): Efficiency ~ age + Sex_M + HelpReceived + Hammer_G + Hammer_Q + Hammer_wood
    Random effect: intercept by chimpanzee

    Returns the fitted MixedLMResults object.
    """
    # Columns used as fixed effects
    fixed_effects = ['age', 'Sex_M', 'HelpReceived', 'Hammer_G', 'Hammer_Q', 'Hammer_wood']

    # Ensure the columns exist in the dataframe
    missing = [c for c in fixed_effects + ['Efficiency', 'chimpanzee'] if c not in df.columns]
    if missing:
        raise ValueError(f"The following required columns are missing from the dataframe: {missing}")

    # Drop rows with missing values in model columns
    model_df = df.dropna(subset=fixed_effects + ['Efficiency', 'chimpanzee']).copy()

    # Prepare endog and exog for MixedLM
    endog = model_df['Efficiency']
    exog = model_df[fixed_effects]
    exog = sm.add_constant(exog, has_constant='add')

    # Fit mixed effects model with random intercept by chimpanzee
    md = sm.MixedLM(endog, exog, groups=model_df['chimpanzee'])
    mdf = md.fit(reml=False)

    # Print and return results object
    print(mdf.summary())
    return mdf


