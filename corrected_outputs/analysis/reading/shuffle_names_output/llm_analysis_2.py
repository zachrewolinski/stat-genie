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
    """
    Transform raw dataset into analysis-ready dataframe with exact column names used in the model.

    Produces columns:
      - ReaderView: binary (1 = Reader View ON, 0 = OFF)
      - NumWords: number of words on the page (from 'dyslexia' raw column)
      - reading_time_ms: reading time in milliseconds (from 'language' if present or computed as adjusted_running_time - scrolling_time)
      - ReadingWPM: words per minute (NumWords / reading_time_minutes)
      - LogReadingWPM: log(ReadingWPM + 1)
      - Dyslexic: binary (1 = dyslexia_bin >= 1, 0 = dyslexia_bin == 0)
      - Age, Gender, Device, Flesch, ScrollingTime_ms

    Drops rows with missing key fields and removes implausible values (zero or negative reading time).
    """
    df = df.copy()

    # Standardize reading-time source: prefer 'language' (documented as reading time without scrolling),
    # otherwise compute as adjusted_running_time - scrolling_time if both available.
    if 'language' in df.columns:
        df['reading_time_ms'] = df['language']
    else:
        # fallback: adjusted_running_time minus scrolling_time (if available)
        if 'adjusted_running_time' in df.columns and 'scrolling_time' in df.columns:
            df['reading_time_ms'] = df['adjusted_running_time'] - df['scrolling_time']
        else:
            # if no reading time available create column of NaNs
            df['reading_time_ms'] = np.nan

    # NumWords: number of words on page. According to dataset schema, 'dyslexia' numeric column holds word count.
    if 'dyslexia' in df.columns:
        df['NumWords'] = pd.to_numeric(df['dyslexia'], errors='coerce')
    elif 'num_words' in df.columns:
        # If actual num_words column exists use that
        df['NumWords'] = pd.to_numeric(df['num_words'], errors='coerce')
    else:
        df['NumWords'] = np.nan

    # ReaderView: canonical binary variable. Prefer categorical 'reader_view' (Y/N), fallback to 'running_time' (0/1)
    def _parse_reader_view(row):
        if 'reader_view' in row and pd.notnull(row['reader_view']):
            try:
                val = str(row['reader_view']).strip()
            except Exception:
                val = ''
            if val.upper() == 'Y' or val == '1':
                return 1
            if val.upper() == 'N' or val == '0':
                return 0
        # fallback to binary running_time if available
        if 'running_time' in row and pd.notnull(row['running_time']):
            try:
                r = float(row['running_time'])
                return 1 if r == 1 else 0
            except Exception:
                return 0
        return np.nan

    df['ReaderView'] = df.apply(_parse_reader_view, axis=1).astype('float')

    # Dyslexic binary derived from dyslexia_bin where 0=no dyslexia, >=1 = dyslexia (including severe)
    if 'dyslexia_bin' in df.columns:
        df['Dyslexic'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
        df['Dyslexic'] = df['Dyslexic'].apply(lambda x: 1 if pd.notnull(x) and x >= 1 else (0 if x == 0 else np.nan)).astype('float')
    else:
        # If not present, but there's a column 'dyslexia' with categorical codes, attempt to use it
        df['Dyslexic'] = np.nan
        if 'dyslexia' in df.columns:
            # if dyslexia seems to be 0/1/2 mapping
            try:
                tmp = pd.to_numeric(df['dyslexia'], errors='coerce')
                df.loc[tmp >= 1, 'Dyslexic'] = 1
                df.loc[tmp == 0, 'Dyslexic'] = 0
            except Exception:
                pass

    # Other covariates: Age, Gender, Device, Flesch, ScrollingTime_ms
    if 'age' in df.columns:
        df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['Age'] = np.nan

    if 'gender' in df.columns:
        df['Gender'] = pd.to_numeric(df['gender'], errors='coerce')
    else:
        df['Gender'] = np.nan

    if 'device' in df.columns:
        df['Device'] = pd.to_numeric(df['device'], errors='coerce')
    else:
        df['Device'] = np.nan

    if 'Flesch_Kincaid' in df.columns:
        df['Flesch'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    elif 'Flesch' in df.columns:
        df['Flesch'] = pd.to_numeric(df['Flesch'], errors='coerce')
    else:
        df['Flesch'] = np.nan

    if 'scrolling_time' in df.columns:
        df['ScrollingTime_ms'] = pd.to_numeric(df['scrolling_time'], errors='coerce')
    else:
        df['ScrollingTime_ms'] = np.nan

    # Compute ReadingWPM: words per minute = NumWords / (reading_time_ms / 1000 / 60)
    # Remove implausible / zero reading times first
    df['reading_time_ms'] = pd.to_numeric(df['reading_time_ms'], errors='coerce')
    # Define a minimum realistic reading time threshold (e.g., at least 0.5 seconds = 500 ms)
    df.loc[df['reading_time_ms'] <= 0, 'reading_time_ms'] = np.nan

    # Compute minutes and WPM
    df['reading_time_min'] = df['reading_time_ms'] / 1000.0 / 60.0
    df['ReadingWPM'] = df['NumWords'] / df['reading_time_min']

    # Clean up infinite / extremely large values: drop where reading_time_min is NaN or ReadingWPM is inf/NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows with missing critical variables
    required_cols = ['ReaderView', 'Dyslexic', 'ReadingWPM']
    df = df.dropna(subset=required_cols)

    # Winsorize ReadingWPM at 1st and 99th percentiles to reduce influence of outliers
    try:
        lower = df['ReadingWPM'].quantile(0.01)
        upper = df['ReadingWPM'].quantile(0.99)
        df['ReadingWPM'] = df['ReadingWPM'].clip(lower, upper)
    except Exception:
        pass

    # Create logged dependent variable
    df['LogReadingWPM'] = np.log(df['ReadingWPM'] + 1.0)

    # Keep only the columns needed for modeling and return
    keep_cols = ['ReaderView', 'Dyslexic', 'ReadingWPM', 'LogReadingWPM', 'NumWords', 'reading_time_ms',
                 'reading_time_min', 'Age', 'Gender', 'Device', 'Flesch', 'ScrollingTime_ms']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression to test whether Reader View improves reading speed for readers with dyslexia.

    Model specification (primary):
      LogReadingWPM ~ ReaderView * Dyslexic + Age + Gender + Device + Flesch + NumWords + ScrollingTime_ms

    Interaction term ReaderView * Dyslexic tests whether the effect of ReaderView differs for dyslexic readers.

    Returns the fitted statsmodels regression results object (with robust standard errors HC3).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns present
    required = ['LogReadingWPM', 'ReaderView', 'Dyslexic', 'Age', 'Gender', 'Device', 'Flesch', 'NumWords', 'ScrollingTime_ms']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f'Missing required columns for modeling: {missing}')

    # Drop rows with NA in model vars
    model_df = df.dropna(subset=required).copy()

    # Convert ReaderView and Dyslexic to numeric (if not already)
    model_df['ReaderView'] = pd.to_numeric(model_df['ReaderView'], errors='coerce')
    model_df['Dyslexic'] = pd.to_numeric(model_df['Dyslexic'], errors='coerce')

    # Formula with interaction
    formula = 'LogReadingWPM ~ ReaderView * Dyslexic + Age + Gender + Device + Flesch + NumWords + ScrollingTime_ms'

    # Fit OLS with robust SE (HC3)
    fit = smf.ols(formula=formula, data=model_df).fit(cov_type='HC3')

    # Return the fitted results object. The caller can call .summary() or inspect parameters.
    return fit


