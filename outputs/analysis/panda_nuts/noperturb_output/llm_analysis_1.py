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
    Transform the raw dataset into the analysis-ready dataframe.

    Steps:
    - Drop rows with essential missing values.
    - Remove sessions with nonpositive duration.
    - Compute efficiency = nuts_opened / seconds.
    - Compute log_efficiency = log(efficiency + small_constant) to handle zeros and skew.
    - Center age (age_c).
    - Create binary encodings for sex (sex_m) and help (help_binary).
    - Ensure chimpanzee and hammer are categorical.

    Returns a dataframe that includes the columns used in the model:
    ['log_efficiency', 'age_c', 'sex_m', 'help_binary', 'hammer', 'chimpanzee']
    """
    df = df.copy()

    # Drop rows missing core variables
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])

    # Ensure valid session durations
    df = df[df['seconds'] > 0]

    # Compute efficiency (nuts opened per second)
    df['efficiency'] = df['nuts_opened'] / df['seconds']

    # Log-transform the efficiency to reduce skew; add small constant to avoid log(0)
    small_constant = 1e-6
    df['log_efficiency'] = np.log(df['efficiency'] + small_constant)

    # Center age to improve interpretability and model convergence
    df['age_c'] = df['age'] - df['age'].mean()

    # Binary encoding for sex: male = 1, female = 0
    df['sex_m'] = df['sex'].astype(str).str.lower().map({'m': 1, 'f': 0})

    # Binary encoding for help: yes ('y') = 1, no ('n' or 'N') = 0
    df['help_binary'] = df['help'].astype(str).str.lower().map({'y': 1, 'n': 0})

    # Drop rows where mappings failed (unexpected categories)
    df = df.dropna(subset=['sex_m', 'help_binary'])

    # Ensure hammer and chimpanzee are categorical for modeling
    df['hammer'] = df['hammer'].astype('category')
    df['chimpanzee'] = df['chimpanzee'].astype('category')

    # Keep only columns necessary for modeling plus some diagnostics columns
    keep_cols = [
        'chimpanzee',
        'age', 'age_c',
        'sex', 'sex_m',
        'help', 'help_binary',
        'hammer',
        'nuts_opened', 'seconds', 'efficiency', 'log_efficiency'
    ]
    # Some of these may not exist if original df had different naming; intersect to be safe
    keep_cols_present = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols_present].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model to estimate the influence of age, sex, and help on nut-cracking efficiency.

    Model specification:
    - Dependent variable: log_efficiency
    - Fixed effects: age_c + sex_m + help_binary + C(hammer) (hammer as categorical control)
    - Random effects: random intercept for chimpanzee to account for repeated sessions per individual

    Returns the fitted mixed-effects model result object.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['log_efficiency', 'age_c', 'sex_m', 'help_binary', 'hammer', 'chimpanzee']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula; C(hammer) will create dummy variables for hammer types
    formula = 'log_efficiency ~ age_c + sex_m + help_binary + C(hammer)'

    # Fit a mixed effects model with random intercepts for each chimpanzee
    md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
    mdf = md.fit(reml=False)

    # Print and return the fitted model
    print(mdf.summary())
    return mdf


