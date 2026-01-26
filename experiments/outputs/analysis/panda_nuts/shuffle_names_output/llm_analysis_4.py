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
    Clean and harmonize columns, compute efficiency measures, and prepare variables for modeling.

    Assumptions based on provided schema where some column descriptions appear shifted:
      - 'seconds' column contains an identifier for the chimpanzee / observation -> rename to 'chimp_id'
      - 'nuts_opened' column actually holds age in years -> rename to 'age_years'
      - 'age' column contains sex (f/m) -> rename to 'sex'
      - 'help' column contains the number of nuts opened -> rename to 'nuts_opened'
      - 'sex' column contains session duration in seconds -> rename to 'session_seconds'
      - 'chimpanzee' column contains whether the chimpanzee received help (y/N) -> rename to 'received_help'

    The function returns a dataframe that includes at minimum the columns used in the analysis/model.
    """
    df = df.copy()

    # Rename columns to their correct semantic meaning inferred from the schema
    rename_map = {
        'seconds': 'chimp_id',          # id
        'nuts_opened': 'age_years',     # actually age
        'age': 'sex',                   # actually sex (f/m)
        'help': 'nuts_opened',          # number of nuts opened
        'sex': 'session_seconds',       # duration (s)
        'chimpanzee': 'received_help'   # whether received help (y/n)
    }
    df = df.rename(columns=rename_map)

    # Ensure required columns exist after renaming; if not, raise informative error
    required_cols = ['chimp_id', 'age_years', 'sex', 'nuts_opened', 'session_seconds', 'received_help', 'hammer']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after renaming/mapping: {missing}")

    # Type conversions
    # chimp_id as string identifier
    df['chimp_id'] = df['chimp_id'].astype(str)

    # numeric conversions for age, session_seconds, nuts_opened
    df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')
    df['session_seconds'] = pd.to_numeric(df['session_seconds'], errors='coerce')
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')

    # Standardize sex values and create binary indicator for male
    df['sex'] = df['sex'].astype(str).str.strip().str.lower()
    # Map common labels to f/m where possible
    df['sex'] = df['sex'].replace({'female': 'f', 'male': 'm', 'fem': 'f', 'masc': 'm'})
    df['sex_m'] = (df['sex'] == 'm').astype(int)

    # Standardize received_help values to binary
    df['received_help'] = df['received_help'].astype(str).str.strip().str.lower()
    df['received_help'] = df['received_help'].replace({'y': 1, 'yes': 1, 'n': 0, 'no': 0, 'na': np.nan})
    # For any other representations like 'true'/'false' or '1'/'0'
    df.loc[df['received_help'].isin(['1', '0']), 'received_help'] = df.loc[df['received_help'].isin(['1', '0']), 'received_help'].astype(int)
    df['received_help'] = pd.to_numeric(df['received_help'], errors='coerce')

    # Compute efficiency: nuts per second
    # Avoid division by zero; set to NaN where session_seconds <= 0 or missing
    df.loc[df['session_seconds'] <= 0, 'session_seconds'] = np.nan
    df['nuts_per_sec'] = df['nuts_opened'] / df['session_seconds']

    # Drop rows with missing critical variables
    df = df.dropna(subset=['nuts_per_sec', 'age_years', 'sex_m', 'received_help'])

    # Log-transform the efficiency measure to stabilize variance (dependent variable)
    # Add a very small epsilon to avoid log(0) in the unlikely event zeros exist
    eps = 1e-8
    df['log_nuts_per_sec'] = np.log(df['nuts_per_sec'] + eps)

    # Ensure hammer is categorical
    df['hammer'] = df['hammer'].astype(str).astype('category')

    # Keep relevant columns for modeling and diagnostics
    keep_cols = [
        'chimp_id', 'age_years', 'sex', 'sex_m', 'received_help', 'hammer',
        'nuts_opened', 'session_seconds', 'nuts_per_sec', 'log_nuts_per_sec'
    ]
    # Some of these columns may not exist if input data differs; intersect with existing
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a linear model predicting log-transformed nuts-per-second as a function of age, sex, and received_help,
    controlling for hammer type. Clustered (robust) standard errors by chimp_id are used to account for repeated
    observations or non-independence within chimpanzees.

    Returns the fitted results object from statsmodels.
    """
    import statsmodels.formula.api as smf

    # Ensure necessary columns are present
    for c in ['log_nuts_per_sec', 'age_years', 'sex_m', 'received_help', 'hammer', 'chimp_id']:
        if c not in df.columns:
            raise ValueError(f"Required column for modeling missing: {c}")

    # Formula: include hammer as a categorical control
    formula = 'log_nuts_per_sec ~ age_years + sex_m + received_help + C(hammer)'

    # Fit OLS and use clustered standard errors by chimp_id
    ols_model = smf.ols(formula, data=df)
    results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['chimp_id']})

    # The returned object is a RegressionResultsWrapper with clustered covariances applied.
    # For alternative specifications, one could fit a mixed-effects model using statsmodels.MixedLM.
    return results


