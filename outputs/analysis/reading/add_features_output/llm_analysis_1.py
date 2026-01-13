from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/add_features_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into an analysis-ready dataframe.
    Steps:
    - Filter out retake trials (retake_trial == 0) to keep first attempts only.
    - Drop rows with missing essential fields used to compute reading speed or core predictors.
    - Compute reading speed in words per minute (WPM) using adjusted_running_time (ms) and num_words.
    - Remove extreme outliers in reading_wpm (outside 1st-99th percentile) to reduce influence of extreme behavior; then log-transform reading_wpm to produce log_wpm.
    - Create a binary english_native_Y indicator from english_native.
    - Coerce categorical fields to category dtype and ensure dyslexia_bin is integer 0/1.
    - Return a dataframe containing only the columns needed for modeling.
    """
    df = df.copy()

    # Keep only first attempts (filter retakes)
    if 'retake_trial' in df.columns:
        df = df[df['retake_trial'] == 0]

    # Drop rows missing essential fields
    required_cols = ['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Ensure adjusted_running_time is positive
    df = df[df['adjusted_running_time'] > 0]

    # Compute reading speed in words per minute
    # adjusted_running_time is in milliseconds; convert to minutes: ms -> minutes = ms / (1000*60)
    df['reading_wpm'] = df['num_words'] * 60000.0 / df['adjusted_running_time']

    # Remove extreme outliers in reading_wpm by 1st-99th percentile trimming
    lower = df['reading_wpm'].quantile(0.01)
    upper = df['reading_wpm'].quantile(0.99)
    df = df[(df['reading_wpm'] >= lower) & (df['reading_wpm'] <= upper)]

    # Log-transform the dependent variable to reduce skew (add small epsilon if needed)
    df['log_wpm'] = np.log(df['reading_wpm'].clip(lower=1e-6))

    # Binary indicator for english native speakers (Y -> 1, else 0). Handle missing gracefully.
    if 'english_native' in df.columns:
        df['english_native_Y'] = df['english_native'].astype(str).str.upper().eq('Y').astype(int)
    else:
        df['english_native_Y'] = 0

    # Ensure dyslexia_bin is integer 0/1
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Coerce categorical covariates
    for cat in ['device', 'education', 'page_id']:
        if cat in df.columns:
            df[cat] = df[cat].astype('category')
        else:
            # if missing, create a placeholder category
            df[cat] = pd.Categorical(pd.Series([None] * len(df)))

    # Keep only columns required for the model
    keep_cols = [
        'uuid',          # participant id for clustering
        'reader_view',
        'dyslexia_bin',
        'log_wpm',
        'reading_wpm',
        'age',
        'device',
        'english_native_Y',
        'Flesch_Kincaid',
        'num_words',
        'page_id',
        'education'
    ]

    # Only retain columns that exist in the dataframe (defensive)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression testing whether Reader View improves reading speed for readers with dyslexia.

    Model specification (primary):
      log_wpm ~ reader_view * dyslexia_bin + age + english_native_Y + Flesch_Kincaid + num_words
                + C(device) + C(education) + C(page_id)

    We cluster standard errors by participant uuid to account for repeated observations.

    Returns the fitted model results object with cluster-robust covariance.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns present
    required = ['log_wpm', 'reader_view', 'dyslexia_bin', 'uuid']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula. We treat device, education, and page_id as categorical fixed effects.
    formula = (
        'log_wpm ~ reader_view * dyslexia_bin + age + english_native_Y + Flesch_Kincaid + num_words '
        '+ C(device) + C(education) + C(page_id)'
    )

    # Fit OLS
    ols_fit = smf.ols(formula, data=df).fit()

    # Cluster-robust standard errors by participant uuid
    # If uuid isn't unique per row (i.e., repeated measures), this accounts for within-subject correlation
    try:
        results = ols_fit.get_robustcov_results(cov_type='cluster', groups=df['uuid'])
    except Exception:
        # Fall back to heteroskedasticity-robust (HC1) if clustering fails
        results = ols_fit.get_robustcov_results(cov_type='HC1')

    # Return the results object (has .summary(), .params, .bse, etc.)
    return results


