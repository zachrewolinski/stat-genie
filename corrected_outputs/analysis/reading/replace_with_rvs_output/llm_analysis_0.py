from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Example read (kept for compatibility; functions operate on any dataframe passed in)
# If running in other environments, callers may ignore this variable and pass their own df.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/replace_with_rvs_output/reading.csv')
except Exception:
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Steps:
    - Drop rows missing essential variables (speed, reader_view, dyslexia_bin, uuid).
    - Ensure types for reader_view, dyslexia_bin, uuid.
    - Create english_native_bin from english_native (Y -> 1, else 0).
    - Convert numeric controls to numeric and drop rows with missing control values.
    - Center continuous controls (age, num_words, Flesch_Kincaid).
    - Create log_speed = log(speed) as the dependent variable to normalize skew.

    Returns a dataframe containing all columns referenced in the model.
    """
    df = df.copy()

    # Ensure required columns exist before dropna to avoid KeyError
    for col in ['speed', 'reader_view', 'dyslexia_bin', 'uuid']:
        if col not in df.columns:
            df[col] = np.nan

    # Essential columns for analysis
    required = ['speed', 'reader_view', 'dyslexia_bin', 'uuid']
    df = df.dropna(subset=required)

    # Ensure types for reader_view and dyslexia_bin (coerce first, then int)
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')

    # Drop rows if coercion introduced NaNs in these required binary columns
    df = df.dropna(subset=['reader_view', 'dyslexia_bin'])

    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Keep uuid as a simple string identifier to avoid categorical code issues
    df['uuid'] = df['uuid'].astype(str)

    # Create english_native_bin: 1 if 'Y', else 0 (handles NaN as 0)
    if 'english_native' in df.columns:
        eng_series = df['english_native'].fillna('N').astype(str).str.upper()
    else:
        eng_series = pd.Series(['N'] * len(df), index=df.index)
    df['english_native_bin'] = eng_series.map(lambda x: 1 if x == 'Y' else 0).astype(int)

    # Convert numeric controls to numeric (coerce errors to NaN)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['age'] = np.nan

    if 'num_words' in df.columns:
        df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    else:
        df['num_words'] = np.nan

    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    else:
        df['Flesch_Kincaid'] = np.nan

    # Ensure retake_trial is numeric binary
    if 'retake_trial' in df.columns:
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Drop rows missing any of the numeric controls needed for the model
    df = df.dropna(subset=['age', 'num_words', 'Flesch_Kincaid'])

    # Center continuous covariates to improve interpretability
    df['age_c'] = df['age'] - df['age'].mean()
    df['num_words_c'] = df['num_words'] - df['num_words'].mean()
    df['Flesch_Kincaid_c'] = df['Flesch_Kincaid'] - df['Flesch_Kincaid'].mean()

    # Dependent variable: log transform reading speed (natural log). Add small constant if zeros present.
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed'])
    df['log_speed'] = np.log(df['speed'] + 1e-8)

    # Some columns (page_id, device) might not exist in certain datasets; ensure they exist and have no NA
    for col in ['page_id', 'device']:
        if col not in df.columns:
            df[col] = 'missing'
        else:
            df[col] = df[col].fillna('missing').astype(str)

    # Final columns to keep for modeling - must match the conceptual variables contract
    keep_cols = [
        'uuid', 'page_id', 'device', 'reader_view', 'dyslexia_bin',
        'speed', 'log_speed', 'age', 'age_c', 'num_words', 'num_words_c',
        'Flesch_Kincaid', 'Flesch_Kincaid_c', 'retake_trial', 'english_native_bin'
    ]

    # Ensure all keep_cols exist in df (create if missing with appropriate defaults)
    for col in keep_cols:
        if col not in df.columns:
            if col in ['reader_view', 'dyslexia_bin', 'retake_trial', 'english_native_bin']:
                df[col] = 0
            elif col in ['log_speed', 'speed', 'age', 'num_words', 'Flesch_Kincaid', 'age_c', 'num_words_c', 'Flesch_Kincaid_c']:
                df[col] = np.nan
            else:
                df[col] = 'missing'

    # Return only the required final columns (preserve current row order)
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects model to estimate whether Reader View differentially affects reading speed for readers with dyslexia.

    Model specification:
    - Dependent variable: log_speed
    - Fixed effects: reader_view, dyslexia_bin, their interaction, age_c, Flesch_Kincaid_c, num_words_c, retake_trial, english_native_bin, device (categorical), page_id (categorical)
    - Random effects: random intercept for uuid to account for repeated measures per participant

    Returns the fitted statsmodels results object (MixedLMResults).
    """
    import statsmodels.formula.api as smf

    # Work on a copy to avoid modifying the caller's dataframe
    df_work = df.copy()

    # Ensure required columns are present; this mirrors the contract
    required_model_cols = [
        'log_speed', 'reader_view', 'dyslexia_bin', 'age_c', 'Flesch_Kincaid_c',
        'num_words_c', 'retake_trial', 'english_native_bin', 'device', 'page_id', 'uuid'
    ]
    # Drop any rows with missing values in variables used by the model to keep alignment between patsy and groups
    df_work = df_work.dropna(subset=required_model_cols)

    # Ensure proper dtypes for categorical and grouping variables
    df_work['device'] = df_work['device'].astype(str)
    df_work['page_id'] = df_work['page_id'].astype(str)
    df_work['uuid'] = df_work['uuid'].astype(str)

    # Ensure binary predictors are integer-coded
    df_work['reader_view'] = pd.to_numeric(df_work['reader_view'], errors='coerce').astype(int)
    df_work['dyslexia_bin'] = pd.to_numeric(df_work['dyslexia_bin'], errors='coerce').astype(int)
    df_work['retake_trial'] = pd.to_numeric(df_work['retake_trial'], errors='coerce').fillna(0).astype(int)
    df_work['english_native_bin'] = pd.to_numeric(df_work['english_native_bin'], errors='coerce').fillna(0).astype(int)

    # Reset index so statsmodels/patsy operate on a clean 0..n-1 index and groups align
    df_work = df_work.reset_index(drop=True)

    # Formula: include interaction between reader_view and dyslexia_bin
    formula = (
        'log_speed ~ reader_view * dyslexia_bin + '
        'age_c + Flesch_Kincaid_c + num_words_c + retake_trial + english_native_bin + '
        'C(device) + C(page_id)'
    )

    # Fit mixed-effects model with random intercept per participant (uuid)
    md = smf.mixedlm(formula, df_work, groups=df_work['uuid'], re_formula='1')

    # Use maximum likelihood (reml=False) for easier comparison if needed
    mdf = md.fit(reml=False)

    # Print brief summary and return full results object
    print(mdf.summary())
    return mdf