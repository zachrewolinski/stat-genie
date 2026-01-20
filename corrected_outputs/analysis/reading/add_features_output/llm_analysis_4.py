from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure essential columns exist; if not, create placeholders so final dataframe has required columns.
    # We will drop rows missing essential measurements below.
    required_cols = [
        'uuid', 'adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin',
        'age', 'device', 'english_native', 'Flesch_Kincaid', 'retake_trial', 'page_id'
    ]
    for col in required_cols:
        if col not in df.columns:
            # For identifiers and categorical fields create sensible defaults; numeric defaults set to NaN so they can be handled appropriately.
            if col == 'uuid':
                # uuid is essential for clustering — leave missing if truly absent; downstream code will raise.
                df['uuid'] = np.nan
            elif col == 'device':
                df['device'] = 'unknown'
            elif col == 'english_native':
                df['english_native'] = np.nan
            elif col == 'page_id':
                df['page_id'] = 'unknown'
            elif col == 'retake_trial':
                df['retake_trial'] = np.nan
            else:
                df[col] = np.nan

    # Coerce numeric columns (where appropriate)
    df['adjusted_running_time'] = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')

    # Normalize reader_view robustly: handle numeric, boolean, and common string encodings
    if 'reader_view' in df.columns:
        df['reader_view'] = df['reader_view'].replace({
            'Y': 1, 'y': 1, 'N': 0, 'n': 0,
            'True': 1, 'False': 0, 'true': 1, 'false': 0,
            True: 1, False: 0
        })
        df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    else:
        df['reader_view'] = np.nan

    # Dyslexia bin: make numeric and conservative mapping to 0/1
    df['dyslexia_bin'] = df['dyslexia_bin'].replace({
        'Y': 1, 'y': 1, 'N': 0, 'n': 0, 'True': 1, 'False': 0, True: 1, False: 0
    })
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')

    # Drop rows missing essential measurements for computing the DV or required by the model
    df = df.dropna(subset=['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin', 'uuid'])

    # Remove non-positive or extremely small adjusted times
    df = df[df['adjusted_running_time'] > 0]

    # Compute reading speed in words per second (adjusted_running_time is in ms)
    df['reading_speed_wps'] = df['num_words'] / (df['adjusted_running_time'] / 1000.0)

    # Remove implausible extreme speeds
    df = df[(df['reading_speed_wps'] > 0.5) & (df['reading_speed_wps'] < 20)].copy()

    # Log-transform reading speed
    df['log_reading_speed'] = np.log(df['reading_speed_wps'])

    # Ensure dyslexia_bin is binary 0/1 integers
    df['dyslexia_bin'] = df['dyslexia_bin'].fillna(0).astype(int)
    df.loc[~df['dyslexia_bin'].isin([0, 1]), 'dyslexia_bin'] = df['dyslexia_bin'].clip(lower=0, upper=1).astype(int)

    # Ensure reader_view is binary 0/1 integers
    df['reader_view'] = df['reader_view'].fillna(0).astype(int)
    df.loc[~df['reader_view'].isin([0, 1]), 'reader_view'] = df['reader_view'].clip(lower=0, upper=1).astype(int)

    # Map english_native to binary; treat missing as 0 conservatively
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].replace({
            'Y': 1, 'y': 1, 'N': 0, 'n': 0, True: 1, False: 0
        })
        df['english_native_bin'] = pd.to_numeric(df['english_native_bin'], errors='coerce').fillna(0).astype(int)
    else:
        df['english_native_bin'] = 0

    # Age: numeric (may remain NaN if missing)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Retake trial: ensure 0/1 integer; if missing assume 0 (not a retake)
    if 'retake_trial' in df.columns:
        df['retake_trial'] = df['retake_trial'].replace({
            'Y': 1, 'y': 1, 'N': 0, 'n': 0, True: 1, False: 0
        })
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Device: categorical with sensible default
    if 'device' in df.columns:
        df['device'] = df['device'].fillna('unknown').astype('category')
    else:
        df['device'] = pd.Series(['unknown'] * len(df), dtype='category')

    # Page id: categorical
    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].fillna('unknown').astype('category')
    else:
        df['page_id'] = pd.Series(['unknown'] * len(df), dtype='category')

    # Flesch_Kincaid: ensure column exists; if missing create NaN numeric column
    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    else:
        df['Flesch_Kincaid'] = np.nan

    # Ensure uuid is present (kept as-is for clustering). If it's not string, convert to string category
    df['uuid'] = df['uuid'].astype(str)

    # Final set of columns required by the model (plus reading_speed_wps for convenience)
    keep_cols = [
        'uuid', 'reader_view', 'dyslexia_bin', 'reading_speed_wps', 'log_reading_speed',
        'age', 'device', 'english_native_bin', 'Flesch_Kincaid', 'num_words', 'retake_trial', 'page_id'
    ]

    # Return only the required columns that we ensure exist
    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    # Work on a copy to avoid modifying the incoming dataframe
    df = df.copy()

    # Build formula (must use exact column names)
    formula = (
        'log_reading_speed ~ reader_view * dyslexia_bin '
        '+ age + C(device) + english_native_bin + Flesch_Kincaid + num_words + retake_trial + C(page_id)'
    )

    # Determine variables required by the model and drop rows with missing values in those variables.
    required_for_model = [
        'log_reading_speed', 'reader_view', 'dyslexia_bin', 'age',
        'device', 'english_native_bin', 'Flesch_Kincaid', 'num_words', 'retake_trial', 'page_id', 'uuid'
    ]
    df_model = df.dropna(subset=required_for_model).copy()

    # Create integer group codes for clustering (statsmodels expects non-negative integer group labels)
    # Use codes derived from the filtered dataframe so they align with the rows actually used in the model.
    group_codes = pd.Categorical(df_model['uuid']).codes

    # Fit OLS on the filtered dataframe and compute cluster-robust standard errors by participant uuid
    ols_mod = smf.ols(formula, data=df_model)
    results = ols_mod.fit(cov_type='cluster', cov_kwds={'groups': group_codes})

    return results