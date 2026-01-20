from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/panda_nuts/shuffle_names_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the dataset's columns to the variables needed for analysis, compute efficiency (nuts per minute),
    create helpful derived columns, and drop rows with missing essential values.

    Expected original columns (as present in the provided dataset):
    - seconds: individual ID (integer)
    - nuts_opened: actually contains age in years in this dataset schema (numeric)
    - age: contains 'f'/'m' codes (sex)
    - hammer: hammer type (categorical)
    - help: number of nuts opened in session (numeric) -> interpreted as nuts_cracked
    - sex: duration of session in seconds (numeric) -> interpreted as duration_sec
    - chimpanzee: whether received help (y/n)
    """
    df = df.copy()

    # Create/clean ID
    df['ID'] = pd.to_numeric(df['seconds'], errors='coerce').astype('Int64')

    # Map/derive semantic columns according to the schema mapping
    df['age_years'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    # 'age' column actually stores sex codes 'f'/'m' in the provided schema
    df['Sex'] = df['age'].astype(str).str.lower().map({'f': 'Female', 'm': 'Male'})

    df['hammer_type'] = df['hammer'].astype(str)

    # 'help' column in the schema is the number of nuts opened in a session
    df['nuts_cracked'] = pd.to_numeric(df['help'], errors='coerce')

    # 'sex' column in the schema is actually session duration in seconds
    df['duration_sec'] = pd.to_numeric(df['sex'], errors='coerce')

    # 'chimpanzee' column indicates whether the focal received help (y/n)
    df['received_help'] = df['chimpanzee'].astype(str).str.lower().map({'y': 1, 'n': 0, 'yes': 1, 'no': 0})

    # Compute efficiency: nuts cracked per minute
    # Avoid division by zero; will drop invalid rows below
    df['Efficiency'] = df['nuts_cracked'] / (df['duration_sec'] / 60.0)

    # Replace infinite values and drop rows with missing essential variables
    df = df.replace([np.inf, -np.inf], np.nan)

    essential_cols = ['Efficiency', 'age_years', 'Sex', 'received_help', 'hammer_type', 'ID']
    df = df.dropna(subset=essential_cols)

    # Cast types for modeling
    df['received_help'] = df['received_help'].astype(int)
    df['Sex'] = df['Sex'].astype('category')
    df['hammer_type'] = df['hammer_type'].astype('category')

    # Add a log-transformed efficiency (useful if distribution is skewed)
    # Add tiny constant to avoid log(0)
    df['LogEfficiency'] = np.log(df['Efficiency'] + 1e-6)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit an OLS model predicting nut-cracking efficiency from age, sex, and receiving help.
    Controls for hammer type and clusters standard errors by individual ID to account for repeated measures.

    Model formula:
      Efficiency ~ age_years + C(Sex) * received_help + C(hammer_type)

    - The interaction C(Sex) * received_help tests whether the effect of receiving help differs by sex.
    - We cluster standard errors by ID.

    Returns the fitted statsmodels results object (with clustered covariance).
    """
    import statsmodels.formula.api as smf

    # ensure model variables exist
    required = ['Efficiency', 'age_years', 'Sex', 'received_help', 'hammer_type', 'ID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    formula = 'Efficiency ~ age_years + C(Sex) * received_help + C(hammer_type)'
    ols_mod = smf.ols(formula, data=df)

    # Fit and cluster standard errors by individual ID to account for repeated observations
    results = ols_mod.fit(cov_type='cluster', cov_kwds={'groups': df['ID']})

    return results


