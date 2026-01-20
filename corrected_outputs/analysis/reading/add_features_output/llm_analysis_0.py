from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling dataframe.

    Created columns (used in modeling):
      - wpm_reading: words per minute computed from num_words and adjusted_running_time (ms)
      - log_wpm: natural log of wpm_reading
      - ReaderView: binary copy of reader_view (int)
      - DyslexiaBin: binary copy of dyslexia_bin (int)
      - english_native_bin: 1 if english_native == 'Y', else 0

    The function also drops rows with missing or invalid values needed for the model.
    """
    df = df.copy()

    # Ensure relevant numeric columns exist and coerce to numeric where appropriate
    df['adjusted_running_time'] = pd.to_numeric(df.get('adjusted_running_time'), errors='coerce')
    df['num_words'] = pd.to_numeric(df.get('num_words'), errors='coerce')
    df['reader_view'] = pd.to_numeric(df.get('reader_view'), errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df.get('dyslexia_bin'), errors='coerce')
    df['age'] = pd.to_numeric(df.get('age'), errors='coerce')
    # If column missing, df.get returns None -> to_numeric gives NaN column
    df['Flesch_Kincaid'] = pd.to_numeric(df.get('Flesch_Kincaid'), errors='coerce')
    df['retake_trial'] = pd.to_numeric(df.get('retake_trial'), errors='coerce')

    # Ensure device column exists and has a default value
    if 'device' not in df.columns:
        df['device'] = 'unknown'
    else:
        # Fill missing device values with 'unknown' and ensure string dtype
        df['device'] = df['device'].fillna('unknown').astype(str)

    # english_native -> binary indicator (be liberal about 'Y' matching)
    if 'english_native' in df.columns:
        eng = df['english_native'].astype(str).str.upper().str.strip()
        df['english_native_bin'] = eng.map({'Y': 1, 'N': 0})
        # Default unexpected/missing to 0 (non-native)
        df['english_native_bin'] = df['english_native_bin'].fillna(0).astype(int)
    else:
        df['english_native_bin'] = 0

    # Clean retake_trial missing values -> assume 0 if missing
    df['retake_trial'] = df['retake_trial'].fillna(0).astype(int)

    # Drop rows missing the core measurement ingredients required for modeling.
    # These include adjusted_running_time, num_words, reader_view, dyslexia_bin, uuid,
    # plus age and Flesch_Kincaid which are explicit covariates in the model.
    required_for_model = [
        'adjusted_running_time',
        'num_words',
        'reader_view',
        'dyslexia_bin',
        'uuid',
        'age',
        'Flesch_Kincaid'
    ]
    # Only include required columns that are present in df to avoid KeyError,
    # but we expect age and Flesch_Kincaid to be present (or else rows will be dropped).
    existing_required = [c for c in required_for_model if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Compute words-per-minute from adjusted running time (ms -> minutes)
    # Prevent division by zero and remove non-positive times
    df = df[df['adjusted_running_time'] > 0]
    df['wpm_reading'] = df['num_words'] / (df['adjusted_running_time'] / 60000.0)

    # Remove infinite / nonpositive wpm values
    df['wpm_reading'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=['wpm_reading'])
    df = df[df['wpm_reading'] > 0]

    # Stabilize skew with natural log transform
    df['log_wpm'] = np.log(df['wpm_reading'])

    # Standardize/cast key modeling columns
    # Ensure ReaderView and DyslexiaBin are integer 0/1
    df['ReaderView'] = df['reader_view'].astype(int)
    df['DyslexiaBin'] = df['dyslexia_bin'].astype(int)

    # Ensure uuid is kept as-is (string)
    df['uuid'] = df['uuid'].astype(str)

    # Keep only columns needed for modeling and downstream diagnostics
    keep_cols = [
        'uuid',
        'page_id',
        'log_wpm',
        'wpm_reading',
        'ReaderView',
        'DyslexiaBin',
        'age',
        'device',
        'english_native_bin',
        'Flesch_Kincaid',
        'num_words',
        'retake_trial'
    ]

    # Some datasets may lack page_id; ensure we don't error
    existing_keep = [c for c in keep_cols if c in df.columns]
    df_out = df[existing_keep].reset_index(drop=True)

    return df_out


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model predicting log-transformed WPM. Key test: interaction between ReaderView and DyslexiaBin.

    Formula: log_wpm ~ ReaderView * DyslexiaBin + age + english_native_bin + Flesch_Kincaid + num_words + retake_trial + C(device)

    Standard errors are clustered by participant (uuid) because participants contribute multiple trials.
    Returns the fitted results object (statsmodels RegressionResultsWrapper).
    """
    # Ensure device exists; if not, create a placeholder (defensive; transform should have done this)
    if 'device' not in df.columns:
        df = df.copy()
        df['device'] = 'unknown'

    formula = (
        'log_wpm ~ ReaderView * DyslexiaBin + age + english_native_bin + '
        'Flesch_Kincaid + num_words + retake_trial + C(device)'
    )

    # Fit the model on the dataframe provided (which should already be the final modeling df).
    mod = smf.ols(formula, data=df)

    # Fit the model and cluster standard errors by participant (uuid)
    # To ensure the groups array lines up with the rows used by the model (i.e., after any row drops),
    # we factorize the uuid column to integer codes and pass those codes as the groups.
    if 'uuid' in df.columns:
        groups = pd.factorize(df['uuid'])[0]
        results = mod.fit(cov_type='cluster', cov_kwds={'groups': groups})
    else:
        results = mod.fit()

    return results