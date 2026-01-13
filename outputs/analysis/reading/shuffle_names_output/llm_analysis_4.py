from typing import Any
import numpy as np
import pandas as pd


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe with:
      - ReadingTime_ms: reading time in milliseconds (primary source: 'language'; fallback: adjusted_running_time - scrolling_time)
      - Words: number of words on the page (primary source: 'dyslexia' column which in this dataset holds word counts; fallback: 'num_words' if available and plausible)
      - ReadingSpeed_wpm: words per minute
      - ReaderView: binary indicator 1 = Reader View ON, 0 = OFF
      - DyslexiaBinary: 0 = No dyslexia, 1 = Dyslexia (includes mild/severe)
      - DyslexiaLevel: categorical label ('No','Dyslexia','Severe') from dyslexia_bin
    It also drops rows with missing/invalid values required for the analysis.
    """
    df = df.copy()

    # ---- Standardize column names used below: expect original columns from schema ----
    # reading time (ms) — the schema indicates 'language' holds reading time (adjusted_running_time - scrolling)
    if 'language' in df.columns:
        df['ReadingTime_ms'] = pd.to_numeric(df['language'], errors='coerce')
    else:
        df['ReadingTime_ms'] = np.nan

    # fallback: use adjusted_running_time - scrolling_time if ReadingTime_ms missing or non-positive
    if 'adjusted_running_time' in df.columns and 'scrolling_time' in df.columns:
        adj = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
        scroll = pd.to_numeric(df['scrolling_time'], errors='coerce')
        fallback_read = adj - scroll
        mask = (df['ReadingTime_ms'].isna()) | (df['ReadingTime_ms'] <= 0)
        df.loc[mask, 'ReadingTime_ms'] = fallback_read[mask]

    # Words on page: in this dataset the 'dyslexia' column appears to contain word counts (values ~100-400)
    if 'dyslexia' in df.columns:
        df['Words'] = pd.to_numeric(df['dyslexia'], errors='coerce')
    else:
        # fallback to 'num_words' if available
        df['Words'] = pd.to_numeric(df.get('num_words', pd.Series([np.nan] * len(df))), errors='coerce')

    # ReaderView indicator: robust parsing of various encodings
    def parse_reader_view(val):
        if pd.isna(val):
            return np.nan
        # If boolean
        if isinstance(val, (bool, np.bool_)):
            return int(val)
        # If numeric or numeric-string
        try:
            num = float(val)
            # treat >0.5 as ON, otherwise OFF
            return 1 if num > 0.5 else 0
        except Exception:
            pass
        # String parsing
        s = str(val).strip().lower()
        if s in {'y', 'yes', 'on', 'true', 't', '1'}:
            return 1
        if s in {'n', 'no', 'off', 'false', 'f', '0'}:
            return 0
        # Unknown encoding -> NaN
        return np.nan

    if 'reader_view' in df.columns:
        df['ReaderView'] = df['reader_view'].apply(parse_reader_view).astype('float')
    else:
        df['ReaderView'] = np.nan

    # Dyslexia status: robust inference from available columns
    if 'dyslexia_bin' in df.columns:
        df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
        df['DyslexiaLevel'] = df['dyslexia_bin'].map({0: 'No', 1: 'Dyslexia', 2: 'Severe'})
        df['DyslexiaBinary'] = df['dyslexia_bin'].apply(lambda x: 1 if pd.notnull(x) and x >= 1 else (0 if x == 0 else np.nan))
    else:
        # Try to infer from a 'dyslexia' textual column if it contains labels
        if 'dyslexia' in df.columns and df['dyslexia'].dtype == object:
            def infer_dyslexia(val):
                if pd.isna(val):
                    return np.nan, np.nan
                s = str(val).strip().lower()
                if s in {'no', 'none', '0', 'n'}:
                    return 0, 'No'
                if any(k in s for k in ['severe']):
                    return 1, 'Severe'
                if any(k in s for k in ['mild', 'dyslexia', 'yes', 'y', '1']):
                    return 1, 'Dyslexia'
                return np.nan, np.nan
            inferred = df['dyslexia'].apply(lambda x: pd.Series(infer_dyslexia(x), index=['DyslexiaBinary', 'DyslexiaLevel']))
            df['DyslexiaBinary'] = inferred['DyslexiaBinary']
            df['DyslexiaLevel'] = inferred['DyslexiaLevel']
        else:
            # If no information, set to NA so rows will be removed downstream (can't infer)
            df['DyslexiaBinary'] = np.nan
            df['DyslexiaLevel'] = np.nan

    # Controls: coerce to numeric where appropriate
    for col in ['device', 'age', 'education', 'Flesch_Kincaid', 'gender', 'scrolling_time', 'adjusted_running_time', 'page_id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Compute ReadingSpeed in words per minute (WPM)
    df['ReadingSpeed_wpm'] = np.nan
    valid_mask = df['ReadingTime_ms'].notna() & (df['ReadingTime_ms'] > 0) & df['Words'].notna() & (df['Words'] > 0)
    df.loc[valid_mask, 'ReadingSpeed_wpm'] = df.loc[valid_mask, 'Words'] / (df.loc[valid_mask, 'ReadingTime_ms'] / 60000.0)

    # Drop rows missing the primary analysis variables
    df = df.dropna(subset=['ReadingSpeed_wpm', 'ReaderView', 'DyslexiaBinary'])

    # Remove implausible speeds (optional cleaning): keep speeds between 5 and 2000 WPM
    df = df[(df['ReadingSpeed_wpm'] >= 5) & (df['ReadingSpeed_wpm'] <= 2000)]

    # Ensure categorical columns are typed for modeling convenience
    df['DyslexiaLevel'] = df['DyslexiaLevel'].astype('category')
    # ReaderView numeric binary
    df['ReaderView'] = df['ReaderView'].astype(float)
    # DyslexiaBinary should be integer 0/1
    # ensure values are 0/1 floats first, then int
    df['DyslexiaBinary'] = pd.to_numeric(df['DyslexiaBinary'], errors='coerce').round().astype(int)

    # Keep only columns needed for modeling and diagnostics
    keep_cols = [
        'ReadingSpeed_wpm', 'ReaderView', 'DyslexiaBinary', 'DyslexiaLevel', 'Words',
        'device', 'age', 'education', 'Flesch_Kincaid', 'gender', 'scrolling_time',
        'adjusted_running_time', 'page_id'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear regression to estimate the effect of ReaderView on reading speed (WPM) and whether
    that effect differs for readers with dyslexia. The primary specification includes an interaction
    between ReaderView and DyslexiaBinary and controls for text length and device/demographics.

    Returns the fitted OLS results object with robust standard errors (HC3), or None if the model
    cannot be fit due to insufficient data.
    """
    import statsmodels.formula.api as smf

    data = df.copy()

    # Ensure required columns exist
    required = ['ReadingSpeed_wpm', 'ReaderView', 'DyslexiaBinary']
    for col in required:
        if col not in data.columns:
            # Required conceptual variables missing: cannot fit model
            return None

    # Drop any rows with NA in the variables that will be used for estimation
    data = data.dropna(subset=required)
    if data.shape[0] < 2:
        # Not enough rows to fit a model
        return None

    # Build formula. DyslexiaBinary is numeric (0/1); ReaderView numeric (0/1).
    controls = []
    for c in ['Words', 'device', 'age', 'education', 'Flesch_Kincaid', 'gender', 'scrolling_time']:
        if c in data.columns:
            controls.append(c)
    controls_part = ' + '.join(controls) if controls else ''

    formula = 'ReadingSpeed_wpm ~ ReaderView * DyslexiaBinary'
    if controls_part:
        formula = formula + ' + ' + controls_part

    # Fit OLS safely
    try:
        model_fit = smf.ols(formula=formula, data=data).fit()
    except Exception:
        return None

    # Produce robust standard errors (HC3) for inference
    try:
        robust_results = model_fit.get_robustcov_results(cov_type='HC3')
    except Exception:
        robust_results = model_fit

    return robust_results