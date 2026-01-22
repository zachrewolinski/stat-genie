from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# NOTE: this file originally read a specific CSV at import time.
# Kept here to preserve original structure; if running in a different
# environment you may want to remove or modify this line.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/noperturb_output/reading.csv')
except Exception:
    df = pd.DataFrame()

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column missing: {c}")

    # Drop obvious missing/invalid measurements
    # adjusted_running_time must be positive and non-null; num_words must be positive
    df = df.dropna(subset=['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin', 'uuid'])
    df = df[df['adjusted_running_time'] > 0]
    df = df[df['num_words'] > 0]

    # Exclude retake trials (they may reflect practice effects)
    if 'retake_trial' in df.columns:
        df = df[df['retake_trial'] == 0]

    # Compute words per minute (wpm) from adjusted_running_time (ms)
    # Avoid divide-by-zero (we already filtered > 0)
    df['wpm'] = df['num_words'] * 60000.0 / df['adjusted_running_time']

    # Filter out extreme implausible speeds (e.g., extremely large due to tiny adjusted_running_time)
    # Use robust bounds: keep wpm between 0.1 and 5000 wpm (very generous). If needed adjust later.
    df = df[(df['wpm'] > 0.1) & (df['wpm'] < 5000)]

    # Log-transform the wpm for modeling (stabilizes skew)
    df['log_wpm'] = np.log(df['wpm'])

    # Ensure ivs and controls have clean types
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').astype(float)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').astype(float)

    # Create english native binary if present
    if 'english_native' in df.columns:
        # Map 'Y'/'N' or other encodings to binary
        df['english_native_binary'] = df['english_native'].map({'Y': 1, 'N': 0})
        # If english_native is not Y/N, try to coerce to numeric
        if df['english_native_binary'].isnull().any():
            df['english_native_binary'] = pd.to_numeric(df['english_native'], errors='coerce')
        # Fill remaining missing english_native with 0 (non-native) only if necessary
        df['english_native_binary'] = df['english_native_binary'].fillna(0).astype(int)
    else:
        # If column not present, create a default column of zeros (conservative)
        df['english_native_binary'] = 0

    # Coerce device and gender into categorical fields
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    else:
        df['device'] = 'unknown'
        df['device'] = df['device'].astype('category')

    if 'gender' in df.columns:
        # If gender is numeric (0/1/2), keep as categorical so model can include C(gender)
        df['gender'] = df['gender'].astype('category')
    else:
        df['gender'] = 'unknown'
        df['gender'] = df['gender'].astype('category')

    # Coerce uuid and page_id to categorical (grouping variables)
    df['uuid'] = df['uuid'].astype('category')
    df['page_id'] = df['page_id'].astype('category')

    # Keep only columns needed for modeling to simplify downstream steps
    keep_cols = [
        'uuid', 'page_id', 'reader_view', 'dyslexia_bin', 'wpm', 'log_wpm',
        'age', 'correct_rate', 'Flesch_Kincaid', 'num_words', 'english_native_binary',
        'device', 'gender'
    ]
    # Add columns that may not be present but referenced (filled with NaN/defaults)
    for extra in keep_cols:
        if extra not in df.columns:
            df[extra] = np.nan

    # Ensure device and gender remain categorical if they were set earlier
    if not pd.api.types.is_categorical_dtype(df['device']):
        df['device'] = df['device'].astype('category')
    if not pd.api.types.is_categorical_dtype(df['gender']):
        df['gender'] = df['gender'].astype('category')

    df = df[keep_cols]

    # Final drop rows with missing values in dependent variable or main ivs/controls
    df = df.dropna(subset=['log_wpm', 'reader_view', 'dyslexia_bin'])

    # Convert reader_view and dyslexia_bin to numeric 0/1
    df['reader_view'] = df['reader_view'].astype(float)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(float)

    # Return transformed dataframe used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects linear model predicting log_wpm with Reader View, Dyslexia, and their interaction.
    Random intercepts are included for participant (uuid) to account for repeated measures.

    Returns the fitted results object. If mixed model fails to converge, falls back to OLS with cluster-robust SEs by uuid.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Check required columns
    required = ['log_wpm', 'reader_view', 'dyslexia_bin', 'uuid', 'device', 'gender']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column for modeling missing: {c}")

    # Build formula: include main IVs + interaction + controls + categorical device and gender
    formula = (
        'log_wpm ~ reader_view * dyslexia_bin '
        '+ correct_rate + age + Flesch_Kincaid + num_words + english_native_binary '
        '+ C(device) + C(gender)'
    )

    # Try Mixed Linear Model with random intercept for uuid
    try:
        md = smf.mixedlm(formula, df, groups=df['uuid'], re_formula='1')
        mdf = md.fit(reml=False, method='lbfgs', maxiter=200)
        return mdf
    except Exception as e:
        # If MixedLM fails (convergence etc.), fall back to OLS with cluster-robust SEs by uuid
        print('MixedLM failed, falling back to OLS with cluster-robust SEs. Error:', e)
        ols_mod = smf.ols(formula, data=df).fit()
        try:
            # Cluster robust standard errors by uuid
            clustered = ols_mod.get_robustcov_results(cov_type='cluster', groups=df['uuid'])
            return clustered
        except Exception:
            # As a last fallback, return the plain OLS fit
            return ols_mod