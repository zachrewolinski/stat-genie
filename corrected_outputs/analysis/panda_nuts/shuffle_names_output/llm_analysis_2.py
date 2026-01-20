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
    Transform the raw dataframe into a cleaned dataframe with the columns required for modeling.

    Assumptions / mapping based on provided schema comments:
    - 'nuts_opened' column in the provided schema actually corresponds to Age (values 3-16).
    - 'sex' column in the provided schema contains session duration in seconds (numeric, e.g., 2.5-135).
    - 'help' column contains the number of nuts opened in the session.
    - 'age' column contains sex categories ('f'/'m').
    - 'chimpanzee' column indicates whether the subject received help ('y'/'n').

    The function creates these final columns (used by the model):
      - Age (numeric)
      - Sex (string categorical)
      - Sex_M (binary 1=male,0=female)
      - NutsOpened (numeric)
      - DurationSec (numeric)
      - ReceivedHelp (binary 1=yes,0=no)
      - Hammer (string categorical)
      - Efficiency (NutsOpened / DurationSec)
      - IndividualID (keeps original 'seconds' column if that encodes ID; otherwise row index)
    """
    df = df.copy()

    # --- Age ---
    # According to the schema notes, 'nuts_opened' contains age values (3-16).
    if 'nuts_opened' in df.columns:
        df['Age'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    else:
        # fallback: if an explicit 'age' numeric column exists
        if 'age' in df.columns and pd.api.types.is_numeric_dtype(df['age']):
            df['Age'] = pd.to_numeric(df['age'], errors='coerce')
        else:
            df['Age'] = np.nan

    # --- Duration (seconds) ---
    # According to schema notes, 'sex' column holds session duration in seconds.
    if 'sex' in df.columns:
        df['DurationSec'] = pd.to_numeric(df['sex'], errors='coerce')
    elif 'seconds' in df.columns:
        df['DurationSec'] = pd.to_numeric(df['seconds'], errors='coerce')
    else:
        df['DurationSec'] = np.nan

    # --- Nuts opened ---
    # According to schema notes, 'help' contains the number of nuts opened in a session.
    if 'help' in df.columns:
        df['NutsOpened'] = pd.to_numeric(df['help'], errors='coerce')
    elif 'nuts_opened' in df.columns:
        # if 'help' missing, fall back to nuts_opened (but per schema this is age)
        df['NutsOpened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    else:
        df['NutsOpened'] = np.nan

    # --- Sex (categorical) ---
    # According to schema, the 'age' column contains sex labels 'f'/'m'.
    if 'age' in df.columns and df['age'].dtype == object:
        # normalize labels
        df['Sex'] = df['age'].astype(str).str.strip().str.lower().map({'f': 'F', 'm': 'M'})
    else:
        # if a dedicated 'sex' categorical column existed it would be used, but in this dataset 'age' stores sex
        df['Sex'] = np.nan

    # Binary sex indicator used in the model
    df['Sex_M'] = (df['Sex'].astype(str).str.upper() == 'M').astype(int)

    # --- Received help ---
    if 'chimpanzee' in df.columns:
        df['ReceivedHelp'] = df['chimpanzee'].astype(str).str.strip().str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    else:
        # If no explicit column present, leave NA (will be dropped later)
        df['ReceivedHelp'] = np.nan

    # coerce to integer 0/1 where possible
    df['ReceivedHelp'] = pd.to_numeric(df['ReceivedHelp'], errors='coerce')

    # --- Hammer control ---
    if 'hammer' in df.columns:
        df['Hammer'] = df['hammer'].astype(str).fillna('Unknown')
    else:
        df['Hammer'] = 'Unknown'

    # --- Individual ID (optional) ---
    # keep the original 'seconds' column as IndividualID if it encodes ID per schema notes
    if 'seconds' in df.columns:
        df['IndividualID'] = df['seconds']
    else:
        df['IndividualID'] = np.arange(len(df)) + 1

    # --- Efficiency (DV): nuts per second ---
    df['Efficiency'] = df['NutsOpened'] / df['DurationSec']

    # Replace infinite and invalid values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows missing the essential variables for the model
    required = ['Efficiency', 'Age', 'Sex_M', 'ReceivedHelp']
    # ReceivedHelp may have been coded as NA; we drop rows missing any required variable
    df = df.dropna(subset=required)

    # Ensure proper dtypes
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['NutsOpened'] = pd.to_numeric(df['NutsOpened'], errors='coerce')
    df['DurationSec'] = pd.to_numeric(df['DurationSec'], errors='coerce')
    df['Efficiency'] = pd.to_numeric(df['Efficiency'], errors='coerce')
    df['ReceivedHelp'] = df['ReceivedHelp'].astype(int)
    df['Sex_M'] = df['Sex_M'].astype(int)

    # Final drop (in case efficiency or Age became NaN after conversions)
    df = df.dropna(subset=['Efficiency', 'Age'])

    # Return the transformed dataframe (contains all columns used by the model)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear regression predicting nut-cracking efficiency (nuts/sec) from Age, Sex, and ReceivedHelp.

    Model specification (OLS):
      Efficiency ~ Age + Sex_M + ReceivedHelp + Age:ReceivedHelp + Hammer (dummy-coded controls)

    Returns the fitted statsmodels results object with robust (HC3) standard errors.
    """
    df = df.copy()

    # Build design matrix
    # Main predictors and an interaction between Age and ReceivedHelp to allow the effect of Age to differ by whether help was received.
    X = pd.DataFrame({
        'Intercept': 1.0,
        'Age': df['Age'],
        'Sex_M': df['Sex_M'],
        'ReceivedHelp': df['ReceivedHelp'],
        'Age_x_ReceivedHelp': df['Age'] * df['ReceivedHelp']
    }, index=df.index)

    # Add hammer type dummy variables (drop-first to avoid multicollinearity)
    hammer_dummies = pd.get_dummies(df['Hammer'].fillna('Unknown'), prefix='Hammer', drop_first=True)
    if not hammer_dummies.empty:
        X = pd.concat([X, hammer_dummies], axis=1)

    # Outcome
    y = df['Efficiency']

    # Fit OLS with robust standard errors (HC3)
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC3')

    # Return the fitted results object
    return results


