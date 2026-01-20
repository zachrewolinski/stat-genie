from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/anonymize_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset (feature1..feature7) into a dataframe ready for modeling.

    Produces the following columns required by the model:
      - ID: int subject identifier (from feature1)
      - Age: numeric age in years (from feature2)
      - Age_c: mean-centered age used in the model
      - Sex: original sex value (from feature3)
      - Sex_male: binary indicator 1 = male, 0 = female
      - HammerType: categorical hammer type (from feature4)
      - NutsOpened: number of nuts opened in session (from feature5)
      - DurationSec: session duration in seconds (from feature6)
      - Help: binary indicator 1 = received help, 0 = no help (from feature7)
      - Efficiency_npm: nuts opened per minute (NutsOpened / DurationSec * 60)
      - log_Efficiency: np.log1p(Efficiency_npm) used as DV

    Drops rows with invalid or missing numeric values (e.g., Duration <= 0).
    """
    df = df.copy()

    # Standardize/rename inputs to descriptive column names
    df['ID'] = pd.to_numeric(df['feature1'], errors='coerce')
    df['Age'] = pd.to_numeric(df['feature2'], errors='coerce')
    df['Sex'] = df['feature3'].astype(str).str.strip()
    df['HammerType'] = df['feature4'].astype(str).str.strip()
    df['NutsOpened'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['DurationSec'] = pd.to_numeric(df['feature6'], errors='coerce')
    df['Help'] = df['feature7'].astype(str).str.strip()

    # Normalize help coding: expect 'y'/'Y' -> 1, others (including 'N') -> 0
    df['Help'] = df['Help'].str.lower().map({'y': 1, 'yes': 1}).fillna(0).astype(int)

    # Sex coding: male = 1, female = 0
    df['Sex_male'] = df['Sex'].str.lower().map({'m': 1, 'male': 1}).fillna(0).astype(int)

    # Remove rows with missing or invalid core measures
    # Duration must be positive to compute rates
    df = df.dropna(subset=['ID', 'Age', 'NutsOpened', 'DurationSec'])
    df = df[df['DurationSec'] > 0]

    # Efficiency: nuts per minute
    df['Efficiency_npm'] = (df['NutsOpened'] / df['DurationSec']) * 60.0

    # Log transform to stabilize variance and handle zeros
    df['log_Efficiency'] = np.log1p(df['Efficiency_npm'])

    # Center age for interpretability
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Ensure HammerType is treated as categorical string
    df['HammerType'] = df['HammerType'].astype(str)

    # Keep only columns necessary for modeling (plus a few useful diagnostics)
    cols_to_keep = [
        'ID', 'Age', 'Age_c', 'Sex', 'Sex_male', 'HammerType',
        'NutsOpened', 'DurationSec', 'Help', 'Efficiency_npm', 'log_Efficiency'
    ]
    df = df.loc[:, cols_to_keep]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects model predicting log_Efficiency from Age_c, Sex_male, Help,
    their interaction (Age_c:Help), and HammerType as a fixed categorical control.
    Include a random intercept for ID to account for repeated measures.

    If a mixed-effects model fails (e.g., only one observation per ID), fall back to OLS.

    Returns the fitted model object (statsmodels result instance).
    """
    import statsmodels.formula.api as smf
    import warnings

    formula = 'log_Efficiency ~ Age_c + Sex_male + Help + Age_c:Help + C(HammerType)'

    # Try mixed-effects model with random intercept for ID
    try:
        # If ID is not integer-like, MixedLM still accepts the groups argument
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            md = smf.mixedlm(formula, data=df, groups=df['ID'])
            mdf = md.fit(reml=False)
        results = mdf
    except Exception as e:
        # Fallback: ordinary least squares
        # This will still give useful fixed-effect estimates if random effects cannot be estimated
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ols_model = smf.ols(formula, data=df).fit()
        results = ols_model

    # Return the fitted results object (caller can inspect .summary(), .params, etc.)
    return results


