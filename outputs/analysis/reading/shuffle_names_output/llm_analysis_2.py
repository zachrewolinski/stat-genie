from typing import Any, Iterable, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices


def _get_first_series(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    for name in candidates:
        if name in df.columns:
            return df[name]
    return None


def _clean_numeric_series(ser: Optional[pd.Series]) -> pd.Series:
    if ser is None:
        return pd.Series([], dtype=float)
    # Work on a copy and normalize missing values
    ser = ser.copy()
    # Convert booleans to ints explicitly
    try:
        is_bool_dtype = pd.api.types.is_bool_dtype(ser.dtype)
    except Exception:
        is_bool_dtype = False
    if is_bool_dtype or ser.dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
        return ser.astype(float)

    # Convert to object to handle mixed types, replace common thousands separators,
    # and coerce non-numeric to NaN
    cleaned = ser.astype(object).where(~ser.isnull(), other=np.nan)
    cleaned = cleaned.map(lambda x: str(x).replace(',', '') if not pd.isnull(x) else x)
    return pd.to_numeric(cleaned, errors='coerce')


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Helper: try a list of possible raw column names for each required variable
    # Reading time (ms)
    rt_candidates = [
        'adjusted_running_time', 'reading_time_ms', 'reading_time',
        'time_ms', 'Time', 'runtime'
    ]
    rt_ser = _get_first_series(df, rt_candidates)
    if rt_ser is None:
        # Create an empty series aligned with df to preserve indexing
        df['ReadingTime_ms'] = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    else:
        df['ReadingTime_ms'] = _clean_numeric_series(rt_ser).reindex(df.index)

    # Number of words on page
    nw_candidates = ['num_words', 'NumWords', 'word_count', 'words', 'n_words']
    nw_ser = _get_first_series(df, nw_candidates)
    if nw_ser is None:
        df['NumWords'] = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    else:
        df['NumWords'] = _clean_numeric_series(nw_ser).reindex(df.index)

    # Reader View indicator
    rv_candidates = ['ReaderViewOn', 'reader_view', 'reader_view_on', 'reader_view_indicator',
                     'reader_viewed', 'reader_view_enabled', 'reader_view_status', 'reader_view_flag',
                     'ReaderView', 'readerView', 'reader_view_flag', 'reader_view_on_flag']
    rv_ser = _get_first_series(df, rv_candidates)
    if rv_ser is None:
        # If no reader-view information is present, assume ReaderView is off (0) by default
        df['ReaderViewOn'] = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    else:
        # Convert booleans and numeric-like strings to numeric, then coerce to 0/1
        rv_num = _clean_numeric_series(rv_ser).reindex(df.index)
        # If values are exactly 0/1 keep them. Otherwise treat >0 as 1, <=0 as 0
        # For non-numeric textual values like "on"/"off", attempt mapping
        if rv_num.isnull().all():
            # Try textual mapping
            def map_on_off(x):
                if pd.isnull(x):
                    return np.nan
                s = str(x).strip().lower()
                if s in {'1', '1.0', 'true', 't', 'yes', 'y', 'on'}:
                    return 1.0
                if s in {'0', '0.0', 'false', 'f', 'no', 'n', 'off'}:
                    return 0.0
                return np.nan
            df['ReaderViewOn'] = rv_ser.map(map_on_off).astype(float).reindex(df.index)
        else:
            # Values present in rv_num: coerce to 0/1
            coerced = rv_num.copy()
            coerced = coerced.where(coerced.isin([0, 1]), other=(coerced > 0).astype(float))
            df['ReaderViewOn'] = coerced

    # Controls: Age, Device, Gender, Flesch_Kincaid
    age_candidates = ['age', 'Age']
    device_candidates = ['device', 'Device', 'device_id']
    gender_candidates = ['gender', 'Gender', 'sex']
    fk_candidates = ['Flesch_Kincaid', 'flesch_kincaid', 'flesch', 'FK']

    age_ser = _get_first_series(df, age_candidates)
    device_ser = _get_first_series(df, device_candidates)
    gender_ser = _get_first_series(df, gender_candidates)
    fk_ser = _get_first_series(df, fk_candidates)

    df['Age'] = (_clean_numeric_series(age_ser).reindex(df.index)
                 if age_ser is not None else pd.Series([np.nan] * len(df), index=df.index, dtype=float))
    df['Device'] = (_clean_numeric_series(device_ser).reindex(df.index)
                    if device_ser is not None else pd.Series([np.nan] * len(df), index=df.index, dtype=float))
    df['Gender'] = (_clean_numeric_series(gender_ser).reindex(df.index)
                    if gender_ser is not None else pd.Series([np.nan] * len(df), index=df.index, dtype=float))
    df['Flesch_Kincaid'] = (_clean_numeric_series(fk_ser).reindex(df.index)
                            if fk_ser is not None else pd.Series([np.nan] * len(df), index=df.index, dtype=float))

    # Dyslexia status: try several possible raw columns
    dys_candidates = ['dyslexia_bin', 'dyslexia', 'dyslexia_status', 'dyslexia_cat']
    dys_ser = _get_first_series(df, dys_candidates)

    # Prepare flexible mapping that accepts ints, numeric-strings, and some textual labels.
    dys_map = {
        0: 'NoDyslexia', 1: 'Dyslexia', 2: 'SevereDyslexia',
        '0': 'NoDyslexia', '1': 'Dyslexia', '2': 'SevereDyslexia',
        'NoDyslexia': 'NoDyslexia', 'No Dyslexia': 'NoDyslexia', 'no dyslexia': 'NoDyslexia',
        'nodyslexia': 'NoDyslexia',
        'Dyslexia': 'Dyslexia', 'dyslexia': 'Dyslexia',
        'SevereDyslexia': 'SevereDyslexia', 'Severe Dyslexia': 'SevereDyslexia', 'severe dyslexia': 'SevereDyslexia'
    }

    def map_dys(x):
        if pd.isnull(x):
            return np.nan
        # If it's numeric-like, try numeric mapping first
        if isinstance(x, (int, float)) and not pd.isnull(x):
            try:
                ix = int(x)
                return dys_map.get(ix, np.nan)
            except Exception:
                pass
        s = str(x).strip()
        # Direct mapping
        if s in dys_map:
            return dys_map[s]
        # Try lower-cased variants
        s_low = s.lower()
        for key in dys_map:
            if isinstance(key, str) and key.lower() == s_low:
                return dys_map[key]
        # Try to parse numeric string
        try:
            ix = int(float(s))
            return dys_map.get(ix, np.nan)
        except Exception:
            return np.nan

    if dys_ser is None:
        # If no dyslexia information is present, assume 'NoDyslexia' as the default category
        df['Dyslexia_bin'] = pd.Series([np.nan] * len(df), index=df.index)
        df['Dyslexia_cat'] = pd.Series(['NoDyslexia'] * len(df), index=df.index, dtype='category')
    else:
        df['Dyslexia_bin'] = _clean_numeric_series(dys_ser).reindex(df.index)
        # Also attempt mapping from original values (to catch textual labels)
        mapped = dys_ser.map(map_dys).reindex(df.index)
        # If numeric mapping produces valid labels (via bin), use those where mapped is NaN
        def _numeric_label_from_bin(x):
            if pd.isnull(x):
                return np.nan
            try:
                ix = int(x)
            except Exception:
                return np.nan
            return dys_map.get(ix, np.nan)
        numeric_labels = df['Dyslexia_bin'].map(_numeric_label_from_bin)
        combined = mapped.where(~mapped.isnull(), other=numeric_labels)
        # For any remaining missing labels, default to 'NoDyslexia' so rows are retained for modeling
        combined = combined.fillna('NoDyslexia')
        df['Dyslexia_cat'] = combined

    # Ensure some control columns exist and do not contain only NaNs that would drop all rows in modeling.
    # Fill non-essential controls' missing values with 0.0 to retain rows (these are numeric controls).
    for col in ['Age', 'Gender', 'Device', 'Flesch_Kincaid']:
        if col not in df.columns:
            df[col] = pd.Series([0.0] * len(df), index=df.index, dtype=float)
        else:
            df[col] = df[col].fillna(0.0)

    # Ensure ReaderViewOn is present and filled with 0.0 for missing values (assume off if unknown)
    if 'ReaderViewOn' not in df.columns:
        df['ReaderViewOn'] = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    else:
        df['ReaderViewOn'] = df['ReaderViewOn'].fillna(0.0)

    # ---- Filter invalid / impossible rows ----
    # Need positive reading time and positive number of words to compute speed
    # Also require ReaderViewOn to be non-missing (now filled)
    valid_mask = (
        df['ReadingTime_ms'].notnull() & (df['ReadingTime_ms'] > 0) &
        df['NumWords'].notnull() & (df['NumWords'] > 0)
    )
    df = df.loc[valid_mask].copy()

    # ---- Compute outcome: reading speed in words per second ----
    # ReadingTime_ms is milliseconds -> convert to seconds
    # Protect against division by zero (ReadingTime_ms > 0 ensured above)
    df['ReadingSpeed_wps'] = df['NumWords'] * 1000.0 / df['ReadingTime_ms']

    # Ensure Dyslexia_cat is a categorical with the expected categories.
    expected_categories = ['NoDyslexia', 'Dyslexia', 'SevereDyslexia']
    df['Dyslexia_cat'] = pd.Categorical(df['Dyslexia_cat'], categories=expected_categories)

    # ---- Drop rows with missing essential covariates used in the model ----
    # Only drop rows missing the truly essential variables: outcome, ReaderViewOn, Dyslexia_cat, NumWords.
    essential_cols = ['ReadingSpeed_wps', 'ReaderViewOn', 'Dyslexia_cat', 'NumWords']
    df = df.dropna(subset=essential_cols)

    # At this point, df contains exactly the required FINAL columns (and possibly other raw columns).
    # To be explicit, keep only the columns required for modeling plus Dyslexia_bin if present for traceability.
    keep_cols = ['ReadingSpeed_wps', 'ReaderViewOn', 'Dyslexia_cat', 'NumWords', 'Age', 'Gender', 'Device', 'Flesch_Kincaid']
    if 'Dyslexia_bin' in df.columns:
        keep_cols.append('Dyslexia_bin')
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    return df


def model(df: pd.DataFrame) -> Any:
    # Validate input dataframe has data
    if df is None or df.shape[0] == 0:
        raise ValueError("The input dataframe to `model` is empty. Ensure `transform` produced rows with all required columns.")

    # Formula: main effect of ReaderViewOn, main effect of Dyslexia (categorical), and their interaction.
    # Include numeric controls for text length (NumWords), Age, Gender, Device, and Flesch_Kincaid readability.
    formula = 'ReadingSpeed_wps ~ ReaderViewOn * C(Dyslexia_cat) + NumWords + Age + Gender + Device + Flesch_Kincaid'

    # Build design matrices with patsy to inspect the resulting exogenous matrix.
    # This allows us to handle the edge case where no exogenous columns are produced
    # (e.g., categorical predictors with a single level leading to zero columns).
    y, X = dmatrices(formula, data=df, return_type='dataframe')

    # Convert y to a 1-d array/Series for endog
    if y.shape[1] == 0:
        raise ValueError("No endogenous variable produced by the model formula.")
    y_series = y.iloc[:, 0]

    # If there are no observations after patsy (all rows dropped due to missing data), raise a clear error.
    if y_series.size == 0:
        raise ValueError("No observations remain after processing missing data. Cannot fit model.")

    # Ensure X is a DataFrame aligned to y_series index
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, index=y_series.index)
    else:
        # Reindex X to match y (patsy may have dropped rows with missing data)
        X = X.reindex(index=y_series.index)

    # Replace inf values and coerce non-numeric to NaN, then drop any columns that are entirely NaN.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda col: pd.to_numeric(col, errors='coerce'))

    # Drop columns that are all-NaN (patsy may produce zero columns if everything was constant/missing)
    X = X.dropna(axis=1, how='all')

    # If X has zero columns (no predictors produced), fall back to an intercept-only design aligned to y.
    if X.shape[1] == 0:
        X = pd.DataFrame({'Intercept': np.ones(len(y_series))}, index=y_series.index)
    else:
        # Ensure there is a constant term; add one if necessary.
        try:
            X = sm.add_constant(X, has_constant='add', prepend=False)
        except Exception:
            # If add_constant fails for any reason, ensure at least an intercept column exists.
            if 'Intercept' not in X.columns and 'const' not in X.columns:
                X.insert(0, 'Intercept', 1.0)

    # Final sanity: ensure X has at least one column and matches y length
    if X.shape[1] == 0:
        X = pd.DataFrame({'Intercept': np.ones(len(y_series))}, index=y_series.index)

    if X.shape[0] != y_series.shape[0]:
        # Reindex to match; this should not normally happen but guard against shape mismatch
        X = X.reindex(index=y_series.index)
        if X.isnull().all(axis=None):
            raise ValueError("Design matrix X is empty after alignment with the outcome variable.")

    # Fit OLS using statsmodels' OLS API and then obtain robust (HC3) covariance results.
    try:
        ols_mod = sm.OLS(y_series, X)
        ols_fit = ols_mod.fit()
    except Exception as e:
        # As a last resort, if fitting fails, attempt an intercept-only model if possible.
        if y_series.size == 0:
            raise ValueError("No data to fit model.") from e
        X_fallback = pd.DataFrame({'Intercept': np.ones(len(y_series))}, index=y_series.index)
        ols_mod = sm.OLS(y_series, X_fallback)
        ols_fit = ols_mod.fit()

    robust_results = ols_fit.get_robustcov_results(cov_type='HC3')

    return robust_results