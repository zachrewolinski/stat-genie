from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/shuffle_names_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure key columns exist; if not, this will raise a KeyError which is useful to surface early
    required = ['reader_view', 'language', 'num_words', 'dyslexia_bin']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Convert numeric-ish columns safely
    df['language'] = pd.to_numeric(df['language'], errors='coerce')  # reading time excluding scrolling (ms)
    df['adjusted_running_time'] = pd.to_numeric(df.get('adjusted_running_time', pd.Series(dtype='float')), errors='coerce')
    df['scrolling_time'] = pd.to_numeric(df.get('scrolling_time', pd.Series(dtype='float')), errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')

    # Drop rows missing essential measurements
    df = df.dropna(subset=['reader_view', 'language', 'num_words', 'dyslexia_bin'])

    # Normalize reader_view to binary: 'Y' -> 1, others -> 0
    df['ReaderView'] = df['reader_view'].astype(str).str.strip().str.upper().map(lambda x: 1 if x == 'Y' else 0).astype(int)

    # Create Dyslexia binary (1 if any dyslexia indicated (dyslexia_bin >= 1), 0 if none)
    # If dyslexia_bin encodes severity as 0/1/2, this collapses severity into presence/absence
    df['Dyslexia'] = (df['dyslexia_bin'] >= 1).astype(int)

    # Remove rows with non-positive or extremely small reading times and non-positive word counts
    # reading time must be > 0; set a conservative lower bound (e.g., > 100 ms) to remove spurious zeros
    df = df[df['language'] > 100]
    df = df[df['num_words'] > 0]

    # Compute reading time in minutes and reading speed in words per minute
    df['ReadingTime_min'] = df['language'] / 60000.0
    # Avoid division by zero (should be handled by filtering above)
    df['ReadingSpeed_wpm'] = df['num_words'] / df['ReadingTime_min']

    # Remove extreme outliers in ReadingSpeed_wpm by trimming the 1st and 99th percentiles
    if df['ReadingSpeed_wpm'].notna().sum() > 0:
        lower = df['ReadingSpeed_wpm'].quantile(0.01)
        upper = df['ReadingSpeed_wpm'].quantile(0.99)
        df = df[(df['ReadingSpeed_wpm'] >= lower) & (df['ReadingSpeed_wpm'] <= upper)].copy()

    # Ensure control variables exist and are numeric where appropriate; coerce missing to NaN
    for col in ['age', 'gender', 'Flesch_Kincaid', 'img_width', 'device', 'scrolling_time']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If a control is missing from the dataset, create it as all-NaN so model function can decide what to do
            df[col] = np.nan

    # Select and return only the columns needed for modeling (keeps intermediate columns for inspection)
    keep_cols = [
        'ReadingSpeed_wpm', 'ReaderView', 'Dyslexia',
        'age', 'gender', 'Flesch_Kincaid', 'img_width', 'device', 'num_words', 'scrolling_time',
        'language', 'ReadingTime_min', 'adjusted_running_time'
    ]
    # Some of these may not exist if not in original df; ensure we only return columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    import statsmodels.formula.api as smf

    # Copy to avoid modifying original
    data = df.copy()

    # Define formula: main effect of ReaderView, moderator Dyslexia (interaction), plus controls
    # Only include controls that exist in the dataframe
    controls = []
    candidate_controls = ['age', 'gender', 'Flesch_Kincaid', 'img_width', 'device', 'num_words', 'scrolling_time']
    for c in candidate_controls:
        if c in data.columns:
            controls.append(c)

    control_str = ' + '.join(controls) if controls else ''
    # Interaction term ReaderView * Dyslexia will test whether the effect of ReaderView differs for people with dyslexia
    if control_str:
        formula = f'ReadingSpeed_wpm ~ ReaderView * Dyslexia + {control_str}'
    else:
        formula = 'ReadingSpeed_wpm ~ ReaderView * Dyslexia'

    # Drop rows with missing values in variables used in the model
    model_vars = ['ReadingSpeed_wpm', 'ReaderView', 'Dyslexia'] + controls
    model_df = data.dropna(subset=model_vars)

    # Fit OLS and use robust (HC3) standard errors to be conservative about heteroskedasticity
    model = smf.ols(formula=formula, data=model_df).fit(cov_type='HC3')

    # Print a brief summary and return the fitted results object
    print(model.summary())
    return model


