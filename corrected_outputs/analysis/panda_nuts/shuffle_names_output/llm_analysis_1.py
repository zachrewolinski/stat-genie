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
    Transform the raw dataset into a cleaned dataframe with the exact columns used in the model.

    Mapping (based on dataset schema notes where column descriptions appear shifted):
      - original 'seconds' column contains subject ID -> subject_id
      - original 'nuts_opened' column contains age in years -> age_years
      - original 'age' column contains sex ('f'/'m') -> sex
      - original 'hammer' stays hammer_type
      - original 'help' column contains number of nuts opened in session -> nuts_opened
      - original 'sex' column contains session duration in seconds -> session_seconds
      - original 'chimpanzee' column indicates whether the subject received help ('y'/'N') -> help_received

    The function returns a dataframe with columns:
      ['subject_id','age_years','sex','hammer_type','nuts_opened','session_seconds','help_received','efficiency_nuts_per_sec']
    """
    df = df.copy()

    # Rename columns to clear, final names
    rename_map = {
        'seconds': 'subject_id',        # actually ID of individual
        'nuts_opened': 'age_years',      # actually age in years (per schema shift)
        'age': 'sex',                    # actually sex (f/m)
        'hammer': 'hammer_type',         # hammer type
        'help': 'nuts_opened',           # actually number of nuts opened in session
        'sex': 'session_seconds',        # actually session duration (seconds)
        'chimpanzee': 'help_received_raw' # raw help indicator (y/N)
    }
    df = df.rename(columns=rename_map)

    # Keep only columns we expect (ignore any extra columns)
    expected = ['subject_id', 'age_years', 'sex', 'hammer_type', 'nuts_opened', 'session_seconds', 'help_received_raw']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        # If columns are missing, raise informative error to the user (rather than silently failing)
        raise ValueError(f"Expected columns not found in input dataframe after rename: {missing}")

    # Coerce types
    # subject_id -> integer (if possible)
    df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce').astype('Int64')

    # Numeric fields
    df['age_years'] = pd.to_numeric(df['age_years'], errors='coerce')
    df['nuts_opened'] = pd.to_numeric(df['nuts_opened'], errors='coerce')
    df['session_seconds'] = pd.to_numeric(df['session_seconds'], errors='coerce')

    # Standardize sex labels (expecting 'f'/'m' or variants). Keep as categorical strings 'F'/'M'.
    df['sex'] = df['sex'].astype(str).str.strip().str.upper().replace({'FEMALE': 'F', 'MALE': 'M'})
    df.loc[~df['sex'].isin(['F', 'M']), 'sex'] = pd.NA

    # Hammer type: string/categorical
    df['hammer_type'] = df['hammer_type'].astype(str).str.strip()

    # Convert help indicator to binary 1/0
    df['help_received'] = df['help_received_raw'].astype(str).str.strip().str.lower().map({
        'y': 1, 'yes': 1, 'true': 1, 't': 1,
        'n': 0, 'no': 0, 'false': 0, 'f': 0
    })
    # Some datasets use 'Y'/'N' or 'y'/'N' etc; map any unmapped values (like 'Y' or 'N') by checking first char
    mask_null_help = df['help_received'].isna() & df['help_received_raw'].notna()
    if mask_null_help.any():
        df.loc[mask_null_help, 'help_received'] = df.loc[mask_null_help, 'help_received_raw'].astype(str).str[0].str.lower().map({'y':1,'n':0})

    # Compute efficiency: nuts opened per second. Guard against division by zero.
    df['efficiency_nuts_per_sec'] = df['nuts_opened'] / df['session_seconds']

    # Drop rows with missing critical values
    critical = ['subject_id', 'age_years', 'sex', 'nuts_opened', 'session_seconds', 'help_received', 'efficiency_nuts_per_sec']
    df = df.dropna(subset=critical)

    # Exclude or warn on non-positive session_seconds (would give inf/zero issues)
    df = df[df['session_seconds'] > 0]

    # Reset index and ensure types are consistent for modeling
    df = df.reset_index(drop=True)
    df['subject_id'] = df['subject_id'].astype(int)
    df['help_received'] = df['help_received'].astype(int)
    df['sex'] = df['sex'].astype('category')
    df['hammer_type'] = df['hammer_type'].astype('category')

    # Return only the final columns used in modeling (explicit ordering)
    final_cols = ['subject_id', 'age_years', 'sex', 'hammer_type', 'nuts_opened', 'session_seconds', 'help_received', 'efficiency_nuts_per_sec']
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear mixed-effects model to estimate how age, sex, and receiving help influence nut-cracking efficiency.

    Model specification (fixed effects):
      efficiency_nuts_per_sec ~ age_years + C(sex) + C(help_received) + C(hammer_type)
      + interaction terms: age_years:C(help_received) and C(sex):C(help_received)

    Random effects:
      Random intercept for subject_id to account for repeated measures / subject-level heterogeneity.

    Returns the fitted model object (statsmodels MixedLMResults) after printing a summary.
    """
    import statsmodels.formula.api as smf

    # Ensure the input dataframe contains the exact columns we expect
    expected = ['subject_id', 'age_years', 'sex', 'hammer_type', 'nuts_opened', 'session_seconds', 'help_received', 'efficiency_nuts_per_sec']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe to model() is missing expected columns: {missing}")

    # Define formula with categorical terms using C(...). Include interactions to test whether help moderates age/sex effects.
    formula = (
        'efficiency_nuts_per_sec ~ age_years + C(sex) + C(help_received) + C(hammer_type) '
        '+ age_years:C(help_received) + C(sex):C(help_received)'
    )

    # Fit mixed effects model with a random intercept for subject_id
    md = smf.mixedlm(formula, df, groups=df['subject_id'])
    mdf = md.fit(reml=False)  # fit via ML for comparability if needed

    # Print and return results
    print(mdf.summary())
    return mdf


