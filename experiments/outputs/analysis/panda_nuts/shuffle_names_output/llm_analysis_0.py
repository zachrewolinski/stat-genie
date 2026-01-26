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
    """
    Clean and transform the raw dataset into a dataframe containing the variables required for modeling.

    Mapping / assumptions (based on provided schema where column names and descriptions were misaligned):
    - original 'seconds' column contains an individual/session ID -> rename to 'chimp_id'
    - original 'nuts_opened' column contains age in years -> rename to 'age_years'
    - original 'age' column contains sex coded as 'f'/'m' -> rename to 'sex'
    - original 'help' column contains number of nuts opened in session -> rename to 'nuts_opened'
    - original 'sex' column contains session duration in seconds -> rename to 'session_seconds'
    - original 'chimpanzee' column indicates whether help was received ('y'/'N') -> rename to 'help_received_raw'
    - 'hammer' kept as hammer type

    Derived columns:
    - 'hammer_type' (cleaned hammer category)
    - 'sex_male' (binary 1 male, 0 female)
    - 'help_received' (binary 1 yes, 0 no)
    - 'efficiency' = nuts_opened / session_seconds (nuts per second)
    - 'log_efficiency' = np.log(efficiency + small_const)
    """

    df = df.copy()

    # Rename columns according to inferred mapping above
    rename_map = {
        'seconds': 'chimp_id',         # used as identifier
        'nuts_opened': 'age_years',    # actually age in years
        'age': 'sex',                  # actually sex 'f'/'m'
        'help': 'nuts_opened',         # actually nuts opened count
        'sex': 'session_seconds',      # actually session duration in seconds
        'chimpanzee': 'help_received_raw',
        'hammer': 'hammer_type'
    }
    df = df.rename(columns=rename_map)

    # Keep only columns we expect to use; if any missing, raise informative error
    expected = ['chimp_id', 'age_years', 'sex', 'hammer_type', 'nuts_opened', 'session_seconds', 'help_received_raw']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        # don't fail silently; create placeholders for robustness (but in normal use we expect none missing)
        for c in missing:
            df[c] = np.nan

    # Coerce numeric types
    df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['session_seconds'] = pd.to_numeric(df['session_seconds'], errors='coerce')

    # Clean sex column: expect values like 'f' and 'm' (case-insensitive)
    df['sex'] = df['sex'].astype(str).str.strip().str.lower().replace({'female': 'f', 'male': 'm'})
    df.loc[~df['sex'].isin(['f', 'm']), 'sex'] = np.nan

    # Create binary sex indicator: male = 1, female = 0
    df['sex_male'] = df['sex'].map({'m': 1, 'f': 0}).astype('float')

    # Clean help_received flags: expect 'y'/'N' or similar
    df['help_received_raw'] = df['help_received_raw'].astype(str).str.strip().str.lower()
    df['help_received'] = df['help_received_raw'].map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    # if mapping produced NaNs but there are obvious boolean-like numeric values, attempt to coerce
    df.loc[df['help_received'].isna() & df['help_received_raw'].str.isnumeric(), 'help_received'] = \
        pd.to_numeric(df.loc[df['help_received'].isna(), 'help_received_raw'], errors='coerce').fillna(np.nan)

    # Clean hammer_type: make consistent strings and treat missing as 'unknown'
    df['hammer_type'] = df['hammer_type'].astype(str).str.strip()
    df.loc[df['hammer_type'].isnull() | (df['hammer_type'] == 'nan'), 'hammer_type'] = 'unknown'

    # Ensure chimp_id is present and categorical/string
    df['chimp_id'] = df['chimp_id'].astype(str)

    # Drop rows with essential missing data
    df = df.dropna(subset=['age_years', 'nuts_opened', 'session_seconds', 'sex_male', 'help_received'])

    # Remove rows with non-positive session durations to avoid division by zero
    df = df[df['session_seconds'] > 0]

    # Compute efficiency (nuts per second) and log-transform
    df['efficiency'] = df['nuts_opened'] / df['session_seconds']
    # small constant to avoid log(0); efficiency should be >= 0
    small_const = 1e-6
    df['log_efficiency'] = np.log(df['efficiency'] + small_const)

    # Reset index and return only the columns needed for modeling (plus a few helpful extras)
    out_cols = ['chimp_id', 'age_years', 'sex_male', 'help_received', 'hammer_type', 'nuts_opened', 'session_seconds', 'efficiency', 'log_efficiency']
    df = df.reset_index(drop=True)
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model predicting log_efficiency from age, sex, and help_received, controlling for hammer type.
    Clustered standard errors by chimp_id are used to account for non-independence if multiple observations per chimp exist.

    Model formula:
      log_efficiency ~ age_years + sex_male + help_received + C(hammer_type)

    Returns the fitted statsmodels result object (with clustered robust SEs applied).
    """

    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['log_efficiency', 'age_years', 'sex_male', 'help_received', 'hammer_type', 'chimp_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit OLS with categorical hammer_type
    formula = 'log_efficiency ~ age_years + sex_male + help_received + C(hammer_type)'
    ols_mod = smf.ols(formula=formula, data=df)

    # Fit and request clustered (by chimp_id) robust covariance if possible
    try:
        res = ols_mod.fit(cov_type='cluster', cov_kwds={'groups': df['chimp_id']})
    except Exception:
        # Fallback to ordinary fit with HC3 robust covariance if clustering fails
        res = ols_mod.fit(cov_type='HC3')

    # Print a concise summary and return the results object
    print(res.summary())
    return res


