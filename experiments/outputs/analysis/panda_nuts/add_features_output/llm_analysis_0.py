from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/add_features_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset to the modeling dataframe.

    Output columns used in the model:
      - Efficiency_per_min: nuts opened per minute (float)
      - Efficiency_log: log(Efficiency_per_min + small_constant) (float)
      - Age_c: centered age (float)
      - Sex_Male: 1 if sex == 'm', 0 if sex == 'f' (int)
      - Help_binary: 1 if help indicates yes, 0 otherwise (int)
      - chimpanzee: ID for random effects (kept as-is)
      - hammer: hammer type (kept as categorical string)
    """

    df = df.copy()

    # Keep only rows with needed numeric values
    df = df.dropna(subset=['nuts_opened', 'seconds', 'age', 'sex', 'help'])

    # Remove non-positive session durations (invalid or zero seconds)
    df = df[df['seconds'] > 0]

    # Compute efficiency: nuts opened per minute
    df['Efficiency_per_min'] = df['nuts_opened'] / df['seconds'] * 60.0

    # Small constant to allow log transform when rate == 0
    small_const = 0.01
    df['Efficiency_log'] = np.log(df['Efficiency_per_min'] + small_const)

    # Create centered age
    df['Age_c'] = df['age'] - df['age'].mean()

    # Create sex binary: 1 = male, 0 = female. Handle capitalization and whitespace
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    df['Sex_Male'] = df['sex'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})
    # If mapping produced NaN for unexpected values, drop those rows
    df = df.dropna(subset=['Sex_Male'])
    df['Sex_Male'] = df['Sex_Male'].astype(int)

    # Create help binary: map common 'yes' tokens to 1, else 0
    df['help'] = df['help'].astype(str).str.strip().str.lower()
    df['Help_binary'] = df['help'].map(lambda x: 1 if x in ['y', 'yes', 'true', 't'] else 0)

    # Ensure chimpanzee id and hammer are present and clean
    df['chimpanzee'] = df['chimpanzee']
    # hammer: keep as categorical string cleaned
    if 'hammer' in df.columns:
        df['hammer'] = df['hammer'].astype(str).str.strip()
    else:
        # If hammer is missing, create a placeholder column to avoid formula errors
        df['hammer'] = 'unknown'

    # Final set of columns used in modeling
    model_cols = ['chimpanzee', 'Age_c', 'Sex_Male', 'Help_binary', 'hammer', 'Efficiency_per_min', 'Efficiency_log']
    df = df[model_cols]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model predicting log efficiency (nuts per minute).

    Model specification:
      Efficiency_log ~ Age_c + Sex_Male + Help_binary + Age_c:Help_binary + Sex_Male:Help_binary + C(hammer)
      random intercepts for chimpanzee

    Returns the fitted model object (statsmodels) and prints a summary.
    """

    import statsmodels.formula.api as smf

    # Ensure categorical hammer treated as categorical in formula using C(hammer)
    formula = 'Efficiency_log ~ Age_c + Sex_Male + Help_binary + Age_c:Help_binary + Sex_Male:Help_binary + C(hammer)'

    # Fit mixed effects model with random intercept for chimpanzee
    # Use reml=False for likelihood-based comparison if needed
    md = smf.mixedlm(formula, data=df, groups=df['chimpanzee'])
    mdf = md.fit(reml=False)

    # Print summary for immediate inspection
    print(mdf.summary())

    # Return the fitted model object for downstream use
    return mdf


