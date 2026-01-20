from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist and make copies to avoid modifying original
    df = df.copy()

    # Standardize presence of dyslexia binary indicator: if dyslexia_bin exists use it, otherwise derive from dyslexia
    if 'dyslexia_bin' not in df.columns:
        if 'dyslexia' in df.columns:
            # treat dyslexia >= 1 as having dyslexia
            df['dyslexia_bin'] = df['dyslexia'].apply(lambda x: 1 if (pd.notnull(x) and x >= 1) else 0)
        else:
            # if no dyslexia info, set to NaN so later dropna will remove these rows
            df['dyslexia_bin'] = np.nan

    # Create ReaderView binary indicator from reader_view
    if 'reader_view' in df.columns:
        # Coerce to numeric, round to nearest integer (handles True/False, 0/1, or numeric), keep NaN as NaN.
        df['ReaderView'] = pd.to_numeric(df['reader_view'], errors='coerce').round()
    else:
        df['ReaderView'] = np.nan

    # Prefer an existing 'speed' measure for reading speed. If missing, attempt to compute words-per-minute
    # from adjusted_running_time and num_words. We'll create ReadingSpeed and then log-transform.
    df['ReadingSpeed'] = np.nan
    if 'speed' in df.columns:
        df.loc[pd.notnull(df['speed']), 'ReadingSpeed'] = pd.to_numeric(df.loc[pd.notnull(df['speed']), 'speed'], errors='coerce')

    # Compute wpm when adjusted_running_time is present and > 0: words / (minutes)
    if 'adjusted_running_time' in df.columns and 'num_words' in df.columns:
        mask = (
            pd.notnull(df['adjusted_running_time'])
            & (pd.to_numeric(df['adjusted_running_time'], errors='coerce') > 0)
            & pd.notnull(df['num_words'])
        )
        # adjusted_running_time is in milliseconds per documentation -> convert to minutes
        computed_wpm = df.loc[mask, 'num_words'].astype(float) / (df.loc[mask, 'adjusted_running_time'].astype(float) / 60000.0)
        df.loc[mask, 'Computed_wpm'] = computed_wpm
        # fill ReadingSpeed where missing
        fill_mask = pd.isnull(df['ReadingSpeed']) & pd.notnull(df.get('Computed_wpm'))
        df.loc[fill_mask, 'ReadingSpeed'] = df.loc[fill_mask, 'Computed_wpm']

    # Remove non-positive or extreme ReadingSpeed entries (invalid trials)
    df.loc[(pd.notnull(df['ReadingSpeed'])) & (df['ReadingSpeed'] <= 0), 'ReadingSpeed'] = np.nan

    # Create LogReadingSpeed using log1p for numerical stability
    df['LogReadingSpeed'] = df['ReadingSpeed'].apply(lambda x: np.log1p(x) if pd.notnull(x) else np.nan)

    # Create EnglishNative binary from english_native column if present
    if 'english_native' in df.columns:
        df['EnglishNative'] = df['english_native'].apply(lambda x: 1 if (pd.notnull(x) and str(x).strip().upper() == 'Y') else 0)
    else:
        df['EnglishNative'] = pd.Series([0] * len(df), index=df.index)

    # Ensure retake_trial exists and is numeric
    if 'retake_trial' in df.columns:
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)
    else:
        df['retake_trial'] = pd.Series([0] * len(df), index=df.index, dtype=int)

    # Center continuous covariates to aid interpretation of main effects
    for col in ['age', 'num_words', 'Flesch_Kincaid']:
        centered = col + '_c'
        if col in df.columns:
            df[centered] = df[col].astype(float) - df[col].astype(float).mean(skipna=True)
        else:
            # create placeholder NaN-centered column
            df[centered] = np.nan

    # Ensure device and page_id and uuid exist
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    else:
        df['device'] = pd.Series(['unknown'] * len(df), index=df.index).astype('category')

    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].astype('category')
    else:
        df['page_id'] = pd.Series(['unknown'] * len(df), index=df.index).astype('category')

    if 'uuid' in df.columns:
        df['uuid'] = df['uuid'].astype('category')
    else:
        # if no uuid, create a synthetic one per row so clustering is disabled but code still runs
        df['uuid'] = pd.Series([f'row_{i}' for i in range(len(df))], index=df.index).astype('category')

    # Filter to rows with the essential variables for the planned analysis
    required = ['LogReadingSpeed', 'ReaderView', 'dyslexia_bin']
    df = df.dropna(subset=required)

    # Ensure ReaderView is numeric int-like (0/1). After dropna it is safe to convert to int.
    # Use standard numpy int to avoid pandas nullable integer dtype which patsy/numpy can misinterpret.
    df['ReaderView'] = df['ReaderView'].astype(int)

    # Convert dyslexia_bin to integer 0/1
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Final columns used in modeling are returned (keeps extras available for diagnostics)
    keep_cols = [
        'uuid', 'page_id', 'ReaderView', 'LogReadingSpeed', 'ReadingSpeed', 'dyslexia_bin',
        'age_c', 'num_words_c', 'Flesch_Kincaid_c', 'device', 'EnglishNative', 'retake_trial'
    ]
    # Keep any that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Formula: main effect of ReaderView, its interaction with dyslexia_bin (moderation),
    # and controls for age, num_words, readability, device, english-native, retake, and page fixed effects.
    # We use LogReadingSpeed as the dependent variable.
    formula = (
        'LogReadingSpeed ~ ReaderView * dyslexia_bin '
        '+ age_c + num_words_c + Flesch_Kincaid_c + EnglishNative + retake_trial '
        '+ C(device) + C(page_id)'
    )

    # Fit OLS
    ols_model = smf.ols(formula=formula, data=df).fit()

    # Compute cluster-robust standard errors clustered by participant (uuid) to account for repeated measures
    try:
        robust = ols_model.get_robustcov_results(cov_type='cluster', groups=df['uuid'])
    except Exception:
        # Fallback to HC3 robust cov if clustering fails
        robust = ols_model.get_robustcov_results(cov_type='HC3')

    # Return the fitted results object with robust covariances. The caller can examine robust.summary().
    return robust