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
    Transform the raw dataset into a cleaned dataframe with the exact columns used in the model.

    The original schema has mismatched/descriptive labels. We perform the following mappings based on the
    dataset description and field summaries:
      - 'seconds' column contains an integer individual ID -> 'ChimpID'
      - 'nuts_opened' column in the raw file actually contains age in years -> 'age_years'
      - 'age' column contains sex categories 'f'/'m' -> 'is_female' (1/0)
      - 'help' column contains the number of nuts opened in the session -> 'nuts_opened'
      - 'sex' column contains session duration in seconds -> 'duration_seconds'
      - 'chimpanzee' column encodes whether the focal individual received help (y/N) -> 'received_help' (1/0)
      - compute efficiency_nuts_per_sec = nuts_opened / duration_seconds

    The final returned dataframe will contain these columns (and 'hammer' carried forward as a control):
      ['ChimpID', 'age_years', 'is_female', 'hammer', 'nuts_opened', 'duration_seconds', 'received_help', 'efficiency_nuts_per_sec']
    """
    df = df.copy()

    # Map columns based on provided schema inconsistencies
    # Chimp ID
    if 'seconds' in df.columns:
        df['ChimpID'] = df['seconds']
    else:
        # if not present, try to create an index-based ID
        df['ChimpID'] = np.arange(len(df)) + 1

    # Age in years (from the column named 'nuts_opened' in the raw schema)
    if 'nuts_opened' in df.columns:
        df['age_years'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    else:
        df['age_years'] = np.nan

    # Sex (raw column named 'age' contains 'f'/'m') -> is_female
    if 'age' in df.columns:
        df['is_female'] = df['age'].astype(str).str.lower().map({'f': 1, 'female': 1, 'm': 0, 'male': 0})
    else:
        df['is_female'] = np.nan

    # Hammer type (keep as-is, convert to category)
    if 'hammer' in df.columns:
        df['hammer'] = df['hammer'].astype('category')
    else:
        df['hammer'] = pd.Categorical([None] * len(df))

    # nuts_opened in session comes from the 'help' column in the raw schema
    if 'help' in df.columns:
        df['nuts_opened'] = pd.to_numeric(df['help'], errors='coerce')
    else:
        df['nuts_opened'] = np.nan

    # session duration in seconds comes from the column named 'sex' in the raw schema
    if 'sex' in df.columns:
        df['duration_seconds'] = pd.to_numeric(df['sex'], errors='coerce')
    else:
        df['duration_seconds'] = np.nan

    # received_help boolean from 'chimpanzee' column (y/N)
    if 'chimpanzee' in df.columns:
        df['received_help'] = df['chimpanzee'].astype(str).str.lower().map({'y': 1, 'yes': 1, 'n': 0, 'no': 0})
    else:
        df['received_help'] = np.nan

    # Drop rows with missing essential numeric variables
    essential_cols = ['age_years', 'is_female', 'nuts_opened', 'duration_seconds', 'received_help']
    df[essential_cols] = df[essential_cols].apply(pd.to_numeric, errors='coerce')

    # Remove rows where duration is missing or <= 0 (can't compute efficiency)
    df = df.dropna(subset=['duration_seconds', 'nuts_opened', 'is_female', 'age_years', 'received_help'])
    df = df[df['duration_seconds'] > 0]

    # Compute efficiency (nuts opened per second)
    df['efficiency_nuts_per_sec'] = df['nuts_opened'] / df['duration_seconds']

    # Keep only the columns necessary for modeling
    final_cols = ['ChimpID', 'age_years', 'is_female', 'hammer', 'nuts_opened', 'duration_seconds', 'received_help', 'efficiency_nuts_per_sec']
    df = df[final_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a mixed-effects linear model predicting efficiency (nuts/sec) from age, sex, and received_help,
    controlling for hammer type and including a random intercept for ChimpID to account for repeated
    observations per individual.

    Model formula:
      efficiency_nuts_per_sec ~ age_years + is_female + received_help + C(hammer)
    Random effects: random intercept for ChimpID

    Returns the fitted MixedLMResults object (and prints a brief summary).
    """
    # Import formula API locally to avoid relying on global imports
    import statsmodels.formula.api as smf

    # Ensure hammer is categorical and has no missing values for modeling
    df = df.copy()
    df['hammer'] = df['hammer'].astype('category')

    # Build formula; C(hammer) treats hammer as categorical
    formula = 'efficiency_nuts_per_sec ~ age_years + is_female + received_help + C(hammer)'

    # Fit mixed-effects model with random intercept for ChimpID
    try:
        md = smf.mixedlm(formula, df, groups=df['ChimpID'])
        mdf = md.fit(reml=False)
    except Exception as e:
        # As a fallback (if MixedLM fails due to small data / singularities), fit OLS with cluster-robust SE
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        ols_mod = smf.ols(formula, data=df).fit()
        # compute cluster-robust (by ChimpID) covariance if ChimpID is available
        try:
            cov = ols_mod.get_robustcov_results(cov_type='cluster', groups=df['ChimpID'])
            print('MixedLM failed; returning OLS with cluster-robust SE. Error was:', e)
            print(cov.summary())
            return cov
        except Exception:
            print('MixedLM failed and cluster-robust OLS failed; returning plain OLS. Error was:', e)
            print(ols_mod.summary())
            return ols_mod

    # Print and return the fitted mixed model
    print(mdf.summary())
    return mdf


