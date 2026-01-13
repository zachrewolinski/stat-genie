from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to a cleaned dataframe containing all columns required for modeling.

    Produces the following final columns (exact names required by downstream analysis):
      - ReaderView: binary (0/1) indicator whether Reader View was active
      - ReadingTime_ms: numeric reading time excluding scrolling
      - Words: number of words on page
      - ReadingSpeed_wpm: derived dependent variable = Words * 60000 / ReadingTime_ms
      - Comprehension: proportion correct (0-1)
      - Age: numeric
      - Device: categorical string
      - Dyslexia: binary indicator (0/1)
      - DyslexiaSeverity: severity code (numeric)
      - Education: categorical string
      - Language: categorical string
      - IsRetake: binary indicator (0/1)
      - NativeEnglish: binary indicator (1 = Y, 0 = N)
      - Gender: categorical string ('Male','Female','Other')

    Basic cleaning steps: renaming, type conversions, drop rows with missing essential values,
    remove non-positive reading times, remove implausible reading speeds (keeps values in a plausible window),
    and simple imputation for comprehension.
    """
    df = df.copy()

    # Rename informative columns from provided schema if present
    rename_map = {
        'feature3': 'ReaderView',
        'feature4': 'TotalTime_ms',
        'feature5': 'ReadingTime_ms',
        'feature6': 'ScrollingTime_ms',
        'feature7': 'Words',
        'feature8': 'Comprehension',
        'feature9': 'ImageWidth',
        'feature10': 'Age',
        'feature11': 'Device',
        'feature12': 'DyslexiaSeverity',
        'feature13': 'Education',
        'feature14': 'GenderCode',
        'feature15': 'Language',
        'feature16': 'IsRetake',
        'feature17': 'Dyslexia',
        'feature18': 'NativeEnglish',
        'feature19': 'FK_readability',
        'feature20': 'feature20'
    }
    df = df.rename(columns=rename_map)

    # Ensure all required final columns exist (create if missing).
    # For numeric/binary columns create NaNs so we can drop rows lacking essentials.
    numeric_required = [
        'ReaderView', 'ReadingTime_ms', 'Words', 'Comprehension',
        'Age', 'Dyslexia', 'DyslexiaSeverity', 'IsRetake', 'NativeEnglish'
    ]
    for col in numeric_required:
        if col not in df.columns:
            df[col] = np.nan

    # For categorical required columns, create default 'Other' if missing
    for col in ['Device', 'Education', 'Language']:
        if col not in df.columns:
            df[col] = 'Other'
    # Gender may be derived from GenderCode below; ensure it exists as placeholder
    if 'Gender' not in df.columns:
        df['Gender'] = 'Other'

    # Convert numeric-like columns safely
    numeric_cols = ['ReaderView', 'ReadingTime_ms', 'TotalTime_ms', 'ScrollingTime_ms', 'Words', 'Comprehension',
                    'Age', 'DyslexiaSeverity', 'IsRetake', 'Dyslexia']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Map binary/categorical fields
    if 'NativeEnglish' in df.columns:
        # Accept 'Y'/'N' and also 1/0
        df['NativeEnglish'] = df['NativeEnglish'].map({'Y': 1, 'N': 0, 'y': 1, 'n': 0}).fillna(df['NativeEnglish'])
        df['NativeEnglish'] = pd.to_numeric(df['NativeEnglish'], errors='coerce')

    # Map GenderCode to readable categorical Gender if present
    if 'GenderCode' in df.columns:
        # Allow numeric or string codes
        gender_map = {0: 'Male', 1: 'Female', 2: 'Other', '0': 'Male', '1': 'Female', '2': 'Other',
                      'M': 'Male', 'F': 'Female', 'Male': 'Male', 'Female': 'Female'}
        df['Gender'] = df['GenderCode'].map(gender_map).fillna(df.get('Gender', 'Other'))
    else:
        # If Gender column already exists ensure it's string
        df['Gender'] = df['Gender'].astype(str).fillna('Other')

    # Ensure Device / Education / Language are strings (categorical)
    for col in ['Device', 'Education', 'Language']:
        if col in df.columns:
            # astype(str) will convert nan to 'nan' so use fillna first
            df[col] = df[col].fillna('Other').astype(str)
        else:
            df[col] = 'Other'

    # Keep only rows with non-missing essential measures: ReadingTime_ms, Words, ReaderView, Dyslexia
    essential = ['ReadingTime_ms', 'Words', 'ReaderView', 'Dyslexia']
    # All essential columns now exist (possibly filled with NaN); drop rows missing any essential measure
    df = df.dropna(subset=essential)

    # Remove non-positive reading times (avoids division by zero)
    # Ensure ReadingTime_ms exists and is numeric due to above steps
    df = df[df['ReadingTime_ms'] > 0]

    # Convert ReaderView and Dyslexia to integer 0/1
    # After dropna they should be convertible to int; coerce just in case
    df['ReaderView'] = pd.to_numeric(df['ReaderView'], errors='coerce').astype(int)
    df['Dyslexia'] = pd.to_numeric(df['Dyslexia'], errors='coerce').astype(int)

    # Compute reading speed (WPM) from Words and ReadingTime_ms (ms -> minutes)
    df['ReadingSpeed_wpm'] = df['Words'] * 60000.0 / df['ReadingTime_ms']

    # Filter implausible reading speeds to reduce influence of measurement errors / bots
    # Keep speeds in a broad plausible window: 10 <= wpm <= 1000
    df = df[(df['ReadingSpeed_wpm'] >= 10) & (df['ReadingSpeed_wpm'] <= 1000)]

    # Impute missing comprehension with the sample mean (simple, conservative choice)
    if 'Comprehension' in df.columns:
        if df['Comprehension'].isnull().any():
            mean_comp = df['Comprehension'].mean()
            # If mean is NaN (all missing), fallback to 0.0
            if np.isnan(mean_comp):
                mean_comp = 0.0
            df['Comprehension'] = df['Comprehension'].fillna(mean_comp)
    else:
        df['Comprehension'] = 0.0

    # Ensure IsRetake, NativeEnglish are numeric 0/1
    if 'IsRetake' in df.columns:
        df['IsRetake'] = df['IsRetake'].fillna(0)
        df['IsRetake'] = pd.to_numeric(df['IsRetake'], errors='coerce').fillna(0).astype(int)
    else:
        df['IsRetake'] = 0
    if 'NativeEnglish' in df.columns:
        df['NativeEnglish'] = df['NativeEnglish'].fillna(0)
        df['NativeEnglish'] = pd.to_numeric(df['NativeEnglish'], errors='coerce').fillna(0).astype(int)
    else:
        df['NativeEnglish'] = 0

    # Ensure DyslexiaSeverity exists (fill 0 if missing)
    if 'DyslexiaSeverity' not in df.columns:
        df['DyslexiaSeverity'] = 0
    else:
        df['DyslexiaSeverity'] = df['DyslexiaSeverity'].fillna(0)
        df['DyslexiaSeverity'] = pd.to_numeric(df['DyslexiaSeverity'], errors='coerce').fillna(0)

    # Final housekeeping: reset index
    df = df.reset_index(drop=True)

    # Ensure final dataframe contains exactly the required conceptual columns (and any helpers)
    # Required columns per contract must be present:
    required_final = [
        'ReaderView', 'ReadingSpeed_wpm', 'Dyslexia', 'DyslexiaSeverity', 'Words',
        'ReadingTime_ms', 'Comprehension', 'Age', 'Device', 'Education', 'Language',
        'IsRetake', 'NativeEnglish', 'Gender'
    ]
    # If any required_final columns are missing at this point (shouldn't be), add sensible defaults
    for col in required_final:
        if col not in df.columns:
            if col in ['Device', 'Education', 'Language', 'Gender']:
                df[col] = 'Other'
            elif col in ['ReaderView', 'Dyslexia', 'IsRetake', 'NativeEnglish', 'DyslexiaSeverity']:
                df[col] = 0
            else:
                df[col] = np.nan

    # Coerce categorical columns to string type to avoid unexpected NA categories
    for col in ['Device', 'Education', 'Language', 'Gender']:
        df[col] = df[col].astype(str).fillna('Other')

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Run two primary analyses to answer the research question:
      1) Primary analysis restricted to readers who have dyslexia (Dyslexia == 1): estimate the effect of ReaderView on ReadingSpeed_wpm controlling for covariates.
      2) Interaction analysis on the full sample: test ReaderView * Dyslexia interaction to see whether ReaderView effect differs by dyslexia status.

    Returns a dictionary with keys:
      - 'dyslexic_model': statsmodels regression result for the dyslexic-only sample (or None if too few rows)
      - 'interaction_model': statsmodels regression result for the full-sample interaction model
    """
    # Work on a copy
    df = df.copy()

    results = {}

    # If the dataframe is empty, cannot fit models
    if df.shape[0] == 0:
        results['dyslexic_model'] = None
        results['interaction_model'] = None
        return results

    # Define common formula terms for controls (use exact column names)
    controls = 'Comprehension + Age + C(Gender) + IsRetake + NativeEnglish + DyslexiaSeverity + C(Device) + C(Education) + C(Language)'

    # 1) Model in dyslexic-only subset
    df_dys = df[df['Dyslexia'] == 1]
    if df_dys.shape[0] < 20:
        # If too few dyslexic observations, return None for this model
        results['dyslexic_model'] = None
    else:
        formula_dys = f'ReadingSpeed_wpm ~ ReaderView + {controls}'
        try:
            dys_model = smf.ols(formula_dys, data=df_dys).fit()
            results['dyslexic_model'] = dys_model
        except Exception:
            # If fitting fails for any reason, return None for this element
            results['dyslexic_model'] = None

    # 2) Interaction model on full sample
    formula_inter = f'ReadingSpeed_wpm ~ ReaderView * Dyslexia + {controls}'
    try:
        interaction_model = smf.ols(formula_inter, data=df).fit()
        results['interaction_model'] = interaction_model
    except Exception:
        results['interaction_model'] = None

    # Return both fitted results; caller can call .summary() on each non-None element
    return results