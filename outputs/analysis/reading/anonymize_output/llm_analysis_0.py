from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from types import SimpleNamespace


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Rename known raw feature columns to canonical intermediate names when present
    rename_map = {
        'feature1': 'ParticipantID',
        'feature2': 'PageID',
        'feature3': 'ReaderView_raw',
        'feature4': 'TotalTime_ms',
        'feature5': 'ReadTime_ms',
        'feature6': 'ScrollTime_ms',
        'feature7': 'Words',
        'feature8': 'ComprehensionRate',
        'feature9': 'ImageWidth',
        'feature10': 'Age',
        'feature11': 'Device',
        'feature12': 'Dyslexia_multiclass',
        'feature13': 'Education',
        'feature14': 'Gender_raw',
        'feature15': 'Language',
        'feature16': 'Retake',
        'feature17': 'Dyslexia',
        'feature18': 'NativeEnglish_raw',
        'feature19': 'Readability',
        'feature20': 'feature20'
    }
    # Only keys that exist will be renamed; pandas.rename ignores absent keys
    df = df.rename(columns=rename_map)

    # Helper to get a Series from the dataframe using several candidate column names.
    # Returns a Series of NaNs (length matches df) if none found.
    def get_series(*candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([np.nan] * len(df), index=df.index)

    # Ensure all intermediate raw columns exist (create from alternatives or as NaN)
    df['ReaderView_raw'] = get_series('ReaderView_raw', 'ReaderView')
    df['ReadTime_ms'] = get_series('ReadTime_ms', 'ReadTime', 'feature4')
    df['Words'] = get_series('Words', 'words', 'feature7')
    df['Dyslexia'] = get_series('Dyslexia', 'feature17', 'Dyslexia_multiclass')
    df['Age'] = get_series('Age', 'feature10')
    df['Device'] = get_series('Device', 'feature11')
    df['Education'] = get_series('Education', 'feature13')
    df['Gender_raw'] = get_series('Gender_raw', 'feature14', 'Gender')
    df['NativeEnglish_raw'] = get_series('NativeEnglish_raw', 'feature18', 'NativeEnglish')
    df['Retake'] = get_series('Retake', 'feature16')
    df['Readability'] = get_series('Readability', 'feature19')
    # Ensure ParticipantID exists and is string; fill missing with 'unknown'
    pid_series = get_series('ParticipantID', 'feature1')
    pid_series = pid_series.fillna('unknown').astype(str)
    df['ParticipantID'] = pid_series

    # Keep only rows with at least some of the core raw variables present.
    # Avoid KeyError by intersecting with existing columns.
    required_raw = ['ReaderView_raw', 'ReadTime_ms', 'Words', 'Dyslexia', 'Age']
    present_required_raw = [c for c in required_raw if c in df.columns]
    if present_required_raw:
        df = df.dropna(subset=present_required_raw)

    # Process ReaderView and Dyslexia as binary ints (0/1)
    # Coerce non-numeric encodings sensibly
    df['ReaderView'] = pd.to_numeric(df['ReaderView_raw'], errors='coerce').fillna(0).astype(int)
    df['Dyslexia'] = pd.to_numeric(df['Dyslexia'], errors='coerce').fillna(0).astype(int)

    # ReadTime_ms: numeric and enforce a minimum floor to avoid division by zero
    df['ReadTime_ms'] = pd.to_numeric(df['ReadTime_ms'], errors='coerce')
    df = df.dropna(subset=['ReadTime_ms'])
    if not df.empty:
        df.loc[df['ReadTime_ms'] < 50, 'ReadTime_ms'] = 50.0  # floor at 50 ms

    # Words numeric and positive
    df['Words'] = pd.to_numeric(df['Words'], errors='coerce')
    df = df.dropna(subset=['Words'])
    if not df.empty:
        df = df[df['Words'] > 0]

    # Compute Words Per Minute (WPM) from words and ReadTime_ms
    # ReadTime_ms is milliseconds -> seconds = ReadTime_ms / 1000
    # WPM = (words / seconds) * 60
    if not df.empty:
        df['ReadingSpeed_wpm'] = df['Words'] / (df['ReadTime_ms'] / 1000.0) * 60.0
    else:
        df['ReadingSpeed_wpm'] = pd.Series(dtype=float)

    # Remove extreme outliers in ReadingSpeed_wpm
    if 'ReadingSpeed_wpm' in df.columns and not df['ReadingSpeed_wpm'].empty:
        df = df[(df['ReadingSpeed_wpm'] > 1) & (df['ReadingSpeed_wpm'] < 2000)]

    # Log-transform reading speed (natural log)
    # Ensure positive values before log
    if 'ReadingSpeed_wpm' in df.columns and not df['ReadingSpeed_wpm'].empty:
        df['log_ReadingSpeed'] = np.log(df['ReadingSpeed_wpm'].astype(float))
    else:
        df['log_ReadingSpeed'] = pd.Series([np.nan] * len(df), index=df.index)

    # Age numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Device, Education as categorical-like strings
    # Fill missing with 'unknown' before astype(str)
    df['Device'] = df['Device'].fillna('unknown').astype(str)
    df['Education'] = df['Education'].fillna('unknown').astype(str)

    # Gender: convert to string categories, fill missing first
    df['Gender'] = df['Gender_raw'].fillna('unknown').astype(str)

    # NativeEnglish: map 'Y'/'N' to 1/0, fall back to numeric coercion otherwise
    native_map = {'Y': 1, 'N': 0, 'y': 1, 'n': 0, 'Yes': 1, 'No': 0}
    df['NativeEnglish'] = df['NativeEnglish_raw'].map(native_map)
    # If mapping gives NaN, try numeric coercion or treat missing as 0
    df['NativeEnglish'] = pd.to_numeric(df['NativeEnglish'], errors='coerce').fillna(
        pd.to_numeric(df['NativeEnglish_raw'], errors='coerce')
    ).fillna(0).astype(int)

    # Retake numeric 0/1
    df['Retake'] = pd.to_numeric(df['Retake'], errors='coerce').fillna(0).astype(int)

    # Readability numeric
    df['Readability'] = pd.to_numeric(df['Readability'], errors='coerce')

    # ParticipantID ensure string and no missing
    df['ParticipantID'] = df['ParticipantID'].fillna('unknown').astype(str)

    # Final filtering: ensure the model-required columns are present and non-missing
    model_cols = [
        'log_ReadingSpeed', 'ReaderView', 'Dyslexia', 'Age', 'Device',
        'Education', 'Gender', 'NativeEnglish', 'Retake', 'Readability', 'ParticipantID'
    ]
    # Only drop rows for columns that exist; this avoids KeyError if some optional pieces are absent.
    present_model_cols = [c for c in model_cols if c in df.columns]
    if present_model_cols:
        df = df.dropna(subset=present_model_cols)

    # Winsorize log_ReadingSpeed at 1st/99th percentiles if possible
    if 'log_ReadingSpeed' in df.columns and not df['log_ReadingSpeed'].empty:
        lower = df['log_ReadingSpeed'].quantile(0.01)
        upper = df['log_ReadingSpeed'].quantile(0.99)
        df['log_ReadingSpeed'] = df['log_ReadingSpeed'].clip(lower=lower, upper=upper)

    # Keep only columns needed for modeling plus some diagnostics
    keep_cols = model_cols + ['ReadingSpeed_wpm', 'Words', 'ReadTime_ms', 'PageID', 'TotalTime_ms', 'ScrollTime_ms', 'ComprehensionRate']
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    # Formula: main effect of ReaderView, moderator Dyslexia (interaction), plus covariates
    formula = (
        'log_ReadingSpeed ~ ReaderView * Dyslexia + Age + C(Device) + C(Education) + C(Gender)'
        ' + NativeEnglish + Retake + Readability'
    )

    # If the dataframe is empty or lacks sufficient rows after removing rows with missing
    # values in variables referenced by the model, return an empty/stub result rather than
    # allowing patsy to fail with cryptic errors.
    required_for_model = [
        'log_ReadingSpeed', 'ReaderView', 'Dyslexia', 'Age', 'Device',
        'Education', 'Gender', 'NativeEnglish', 'Retake', 'Readability', 'ParticipantID'
    ]
    present_required = [c for c in required_for_model if c in df.columns]
    if not present_required:
        # Nothing to model on
        return SimpleNamespace(params=pd.Series(dtype=float), bse=pd.Series(dtype=float),
                               pvalues=pd.Series(dtype=float), rsquared=np.nan, nobs=0)

    # Drop rows with missing values in the required columns for modeling / clustering
    df_model = df.dropna(subset=present_required)
    # If no data remains, return an empty/stub result
    if df_model.empty:
        return SimpleNamespace(params=pd.Series(dtype=float), bse=pd.Series(dtype=float),
                               pvalues=pd.Series(dtype=float), rsquared=np.nan, nobs=0)

    # Fit OLS
    ols_res = smf.ols(formula, data=df_model).fit()

    # Attempt cluster-robust SEs by ParticipantID; fallback to HC3 if clustering fails
    try:
        clustered = ols_res.get_robustcov_results(cov_type='cluster', groups=df_model['ParticipantID'])
    except Exception:
        clustered = ols_res.get_robustcov_results(cov_type='HC3')

    return clustered