from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


def _to_binary_series(s: pd.Series) -> pd.Series:
    """
    Coerce a series into numeric 0/1 values when possible.
    Recognizes common string encodings like 'Y'/'N', 'yes'/'no', 'on'/'off',
    'true'/'false' (case-insensitive), and numeric 0/1.
    Returns a float dtype series with np.nan where coercion isn't possible.
    """
    if s is None:
        return pd.Series(dtype=float)

    # If already numeric-like, try numeric conversion first
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce').astype(float)

    # Map common categorical encodings
    mapping = {
        'y': 1, 'yes': 1, 'true': 1, 't': 1, 'on': 1, '1': 1,
        'n': 0, 'no': 0, 'false': 0, 'f': 0, 'off': 0, '0': 0
    }
    # Normalize strings and map
    s_str = s.astype(str).str.strip().str.lower()
    mapped = s_str.map(mapping)

    # For any values not mapped, attempt numeric coercion
    not_mapped_mask = mapped.isna()
    if not_mapped_mask.any():
        numeric_attempt = pd.to_numeric(s[not_mapped_mask], errors='coerce')
        mapped.loc[not_mapped_mask] = numeric_attempt

    return mapped.astype(float)


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/noperturb_output/reading.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Creates the following columns used in the model:
      - reading_time_s: adjusted_running_time in seconds
      - reading_speed_wps: num_words / reading_time_s (words per second)
      - log_reading_speed: natural log of reading_speed_wps (dependent variable)
      - age_c: mean-centered age
      - num_words_c: mean-centered num_words
      - Flesch_Kincaid_c: mean-centered Flesch_Kincaid
      - english_native_bin: 1 if english_native == 'Y', 0 if 'N' (na left as np.nan)

    Rows with missing or invalid adjusted_running_time or num_words are dropped.
    Extreme outliers in log_reading_speed (outside the 1st-99th percentile) are trimmed.
    The final dataframe includes all required final columns (some may be entirely NaN).
    """

    # Work on a copy
    df = df.copy()

    # Ensure key columns exist in the raw data; uuid, page_id, device must be present per contract
    required_cols = ['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id', 'device']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns for transform: {}'.format(missing))

    # Drop rows with missing adjusted_running_time or num_words or reader_view or dyslexia_bin
    df = df.dropna(subset=['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin'])

    # Coerce reader_view to binary numeric 0/1
    df['reader_view'] = _to_binary_series(df['reader_view'])

    # Coerce dyslexia_bin to binary numeric 0/1
    df['dyslexia_bin'] = _to_binary_series(df['dyslexia_bin'])

    # If coercion produced NaNs for these critical vars, drop those rows
    df = df.dropna(subset=['reader_view', 'dyslexia_bin'])

    # Convert adjusted_running_time from milliseconds to seconds
    df['reading_time_s'] = pd.to_numeric(df['adjusted_running_time'], errors='coerce') / 1000.0

    # Remove non-positive or missing reading times to avoid division by zero / negative
    df = df[df['reading_time_s'].notna() & (df['reading_time_s'] > 0)]

    # Compute reading speed in words per second
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    df = df[df['num_words'].notna()]
    df['reading_speed_wps'] = df['num_words'].astype(float) / df['reading_time_s'].astype(float)

    # Remove non-positive or infinite speeds
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['reading_speed_wps'])
    df = df[df['reading_speed_wps'] > 0]

    # Log-transform dependent variable to reduce skew
    df['log_reading_speed'] = np.log(df['reading_speed_wps'].astype(float))

    # Trim extreme outliers in the DV to reduce influence of measurement errors (1st-99th percentile)
    lower = df['log_reading_speed'].quantile(0.01)
    upper = df['log_reading_speed'].quantile(0.99)
    df = df[(df['log_reading_speed'] >= lower) & (df['log_reading_speed'] <= upper)]

    # Mean-center continuous controls
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        if df['age'].notna().any():
            df['age_c'] = df['age'] - df['age'].mean()
        else:
            df['age_c'] = np.nan
    else:
        df['age_c'] = np.nan

    # num_words_c (should exist since num_words used above)
    df['num_words_c'] = df['num_words'].astype(float) - df['num_words'].astype(float).mean()

    # Flesch_Kincaid_c
    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
        if df['Flesch_Kincaid'].notna().any():
            df['Flesch_Kincaid_c'] = df['Flesch_Kincaid'] - df['Flesch_Kincaid'].mean()
        else:
            df['Flesch_Kincaid_c'] = np.nan
    else:
        df['Flesch_Kincaid_c'] = np.nan

    # Binary indicator for native English speaker
    if 'english_native' in df.columns:
        df['english_native_bin'] = _to_binary_series(df['english_native'])
    else:
        df['english_native_bin'] = np.nan

    # Ensure retake_trial is numeric 0/1 if present; default to 0 if not present
    if 'retake_trial' in df.columns:
        df['retake_trial'] = _to_binary_series(df['retake_trial'])
        # If conversion produced NaN for some rows, treat those as 0 (assume not a retake)
        df['retake_trial'] = df['retake_trial'].fillna(0.0)
    else:
        df['retake_trial'] = 0.0

    # Ensure device and page_id and uuid exist (they were required above). Keep them as-is.
    # Ensure dtype consistency
    df['uuid'] = df['uuid'].astype(str)
    df['page_id'] = df['page_id'].astype(str)
    df['device'] = df['device'].astype(str)

    # Replace any remaining inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Prepare final columns required by the contract (must exist even if all-NaN)
    keep_cols = [
        'uuid', 'page_id', 'device', 'reader_view', 'dyslexia_bin',
        'log_reading_speed', 'age_c', 'num_words_c', 'Flesch_Kincaid_c',
        'english_native_bin', 'retake_trial'
    ]

    # Ensure all required final columns exist; create with NaN if missing
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Select final columns in the specified order
    df_final = df[keep_cols].reset_index(drop=True)

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model of log_reading_speed on reader_view with an interaction for dyslexia_bin,
    controlling for covariates and including page and device fixed effects. Cluster standard
    errors by participant uuid.

    Model specification (in plain terms):
      log_reading_speed ~ reader_view + dyslexia_bin + reader_view:dyslexia_bin
                         + age_c + num_words_c + Flesch_Kincaid_c + english_native_bin + retake_trial
                         + device dummies + page dummies

    Returns the fitted statsmodels RegressionResultsWrapper (with cluster-robust SEs).
    """

    # Copy to avoid modifying original
    df = df.copy()

    # Ensure required columns exist in the final dataframe
    required = ['log_reading_speed', 'reader_view', 'dyslexia_bin', 'uuid']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns for modeling: {}'.format(missing))

    # Make sure numeric conversions are applied where appropriate
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
    df['log_reading_speed'] = pd.to_numeric(df['log_reading_speed'], errors='coerce')

    # Interaction term (helper column)
    df['reader_view_x_dyslexia'] = df['reader_view'] * df['dyslexia_bin']

    # Base design matrix columns
    X_cols = ['reader_view', 'dyslexia_bin', 'reader_view_x_dyslexia']

    # Add other controls only if they exist AND are not entirely missing
    optional_controls = ['age_c', 'num_words_c', 'Flesch_Kincaid_c', 'english_native_bin', 'retake_trial']
    for c in optional_controls:
        if c in df.columns and df[c].notna().any():
            # Ensure numeric conversion
            df[c] = pd.to_numeric(df[c], errors='coerce')
            # Only include if after conversion there is at least one non-NaN value
            if df[c].notna().any():
                X_cols.append(c)

    # Create device dummies and page dummies (drop first to avoid multicollinearity)
    # Only create dummies when the column exists and there's at least one non-missing value
    if 'device' in df.columns and df['device'].notna().any():
        device_dummies = pd.get_dummies(df['device'].astype(str), prefix='device', drop_first=True)
        for col in device_dummies.columns:
            df[col] = device_dummies[col]
            X_cols.append(col)

    if 'page_id' in df.columns and df['page_id'].notna().any():
        page_dummies = pd.get_dummies(df['page_id'].astype(str), prefix='page', drop_first=True)
        for col in page_dummies.columns:
            df[col] = page_dummies[col]
            X_cols.append(col)

    # Prepare model dataframe containing y and X columns
    # Replace infs with NaN then drop rows with any NaN in predictors or outcome
    df_model = pd.concat([df['log_reading_speed'], df[X_cols]], axis=1)
    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()

    if df_model.shape[0] == 0:
        raise ValueError('No observations remain after dropping rows with missing data for modeling.')

    # Align groups (uuid) with the rows kept
    groups = df.loc[df_model.index, 'uuid'].astype(str)

    # Final X and y
    X = df_model[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df_model['log_reading_speed'].astype(float)

    # Fit OLS with clustered standard errors by uuid
    model_sm = sm.OLS(y, X)
    results = model_sm.fit(cov_type='cluster', cov_kwds={'groups': groups.values})

    return results