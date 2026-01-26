from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/panda_nuts/shuffle_names_output/panda_nuts.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Rename columns according to the schema mapping inferred from descriptions:
    # - 'nuts_opened' column actually contains age (years)
    # - 'age' column contains sex ('f'/'m')
    # - 'help' column contains number of nuts opened in the session
    # - 'sex' column contains duration of the session in seconds
    # - 'chimpanzee' column indicates whether help was received ('y'/'N')
    # - 'seconds' appears to be a small integer id -> SessionID
    df = df.rename(columns={
        'seconds': 'SessionID',
        'nuts_opened': 'Age',
        'age': 'Sex',
        'hammer': 'Hammer',
        'help': 'NutsOpened',
        'sex': 'DurationSec',
        'chimpanzee': 'ReceivedHelp'
    })

    # Normalize and coerce types
    # Sex -> binary (male=1, female=0)
    df['Sex'] = df['Sex'].astype(str).str.strip().str.lower()
    df['SexBinary'] = df['Sex'].map({'m': 1, 'male': 1, 'f': 0, 'female': 0})

    # ReceivedHelp -> binary (yes=1, no=0)
    df['ReceivedHelp'] = df['ReceivedHelp'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})

    # Numeric conversions for Age, NutsOpened, DurationSec
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['NutsOpened'] = pd.to_numeric(df['NutsOpened'], errors='coerce')
    df['DurationSec'] = pd.to_numeric(df['DurationSec'], errors='coerce')

    # Drop rows with missing essential values
    df = df.dropna(subset=['Age', 'SexBinary', 'ReceivedHelp', 'NutsOpened', 'DurationSec'])

    # Remove non-positive durations to avoid division errors
    df = df[df['DurationSec'] > 0]

    # Compute raw efficiency (nuts per second) and log transform
    df['Efficiency'] = df['NutsOpened'] / df['DurationSec']

    # Remove non-positive efficiencies (if any)
    df = df[df['Efficiency'] > 0]

    # Log-transform to stabilize variance and better meet linear model assumptions
    df['LogEfficiency'] = np.log(df['Efficiency'])

    # Keep only the columns required for modeling (and a session id for reference)
    keep_cols = [
        'SessionID',
        'Age',
        'SexBinary',
        'Hammer',
        'NutsOpened',
        'DurationSec',
        'ReceivedHelp',
        'Efficiency',
        'LogEfficiency'
    ]

    # Some datasets may contain additional unexpected columns — ensure returned dataframe contains the model columns
    df_out = df.loc[:, [c for c in keep_cols if c in df.columns]].reset_index(drop=True)

    return df_out


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear model predicting log-efficiency (log nuts/sec) from Age, Sex, and ReceivedHelp.
    Include interactions of ReceivedHelp with Age and Sex to test whether help moderates the effects of age or sex.
    Control for Hammer type as a categorical variable.

    Returns the fitted statsmodels OLS result object (with robust HC3 standard errors).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['LogEfficiency', 'Age', 'SexBinary', 'ReceivedHelp', 'Hammer']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula: main effects + interactions ReceivedHelp*Age and ReceivedHelp*SexBinary + categorical Hammer
    formula = ('LogEfficiency ~ Age + SexBinary + ReceivedHelp '
               '+ Age:ReceivedHelp + SexBinary:ReceivedHelp + C(Hammer)')

    # Fit OLS with robust standard errors (HC3)
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object for inspection (model.summary() can be called by the user)
    return model


