from typing import Any, Dict, List
import re

import numpy as np
import pandas as pd


def _normalize_colname(name: str) -> str:
    # Lowercase, remove non-alphanumeric characters for loose matching
    return re.sub(r'[^0-9a-z]', '', str(name).lower())


def _find_column(df: pd.DataFrame, targets: List[str]) -> str:
    """
    Return the first column name from df.columns that loosely matches any of
    the strings in targets. Matching is case-insensitive and ignores
    non-alphanumeric characters.
    """
    norm_to_col = { _normalize_colname(col): col for col in df.columns }
    for t in targets:
        nt = _normalize_colname(t)
        if nt in norm_to_col:
            return norm_to_col[nt]
    # fallback: if any column's normalized name contains the target number (e.g., '3')
    for t in targets:
        digits = ''.join(ch for ch in t if ch.isdigit())
        if not digits:
            continue
        for nc, orig in norm_to_col.items():
            if digits in nc:
                return orig
    # no match
    return ""


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.

    - Computes time-per-word in seconds and its log (LogTimePerWord).
    - Creates clean/explicit column names used in the model.
    - Drops rows with invalid or missing timing or word counts.
    - Maps categorical variables to consistent string categories where appropriate.

    Returns the transformed dataframe containing all columns listed in the conceptual variables.
    """
    df = df.copy()

    # If the dataframe already contains the final/model columns, assume it's already transformed.
    model_cols = ['LogTimePerWord', 'ReaderView', 'Dyslexia', 'Age', 'Device', 'Education', 'Gender',
                  'NativeEnglish', 'Retake', 'Readability', 'PageID', 'ComprehensionRate']
    aux_cols = ['DyslexiaSeverity', 'ReadingTime_ms', 'Words', 'feature1']
    final_cols = model_cols + aux_cols

    if set(model_cols).issubset(set(df.columns)):
        # Ensure types are reasonable, then return only the kept columns if present
        keep_cols = [c for c in final_cols if c in df.columns]
        return df[keep_cols].copy()

    # Otherwise, attempt to locate raw feature columns (flexible matching).
    # Expected raw features numbers we want to map from:
    expected_features = {
        'feature3': ['feature3', 'feature_3', 'feature 3', 'f3', 'feat3', 'readerview', 'reader_view', 'reader view', 'reader'],
        'feature5': ['feature5', 'feature_5', 'feature 5', 'f5', 'readingtime', 'reading_time', 'reading time',
                     'runningtime', 'running_time', 'running time', 'adjusted_running_time', 'adjustedrunningtime', 'speed', 'time_ms', 'time'],
        'feature7': ['feature7', 'feature_7', 'feature 7', 'f7', 'words', 'wordcount', 'word_count', 'num_words', 'numwords', 'nwords'],
        'feature10': ['feature10', 'feature_10', 'feature 10', 'f10', 'age', 'participant_age'],
        'feature11': ['feature11', 'feature_11', 'feature 11', 'f11', 'device', 'user_device'],
        'feature12': ['feature12', 'feature_12', 'feature 12', 'f12', 'dyslexiaseverity', 'severity'],
        'feature16': ['feature16', 'feature_16', 'feature 16', 'f16', 'retake', 'retake_trial'],
        'feature17': ['feature17', 'feature_17', 'feature 17', 'f17', 'dyslexia', 'dyslexia_bin'],
        'feature18': ['feature18', 'feature_18', 'feature 18', 'f18', 'nativeenglish', 'native_english', 'english_native', 'language_native'],
        'feature19': ['feature19', 'feature_19', 'feature 19', 'f19', 'readability', 'fleschkincaid', 'flesch_kincaid', 'flesch-kincaid', 'flesch kincaid', 'Flesch_Kincaid'],
        'feature2': ['feature2', 'feature_2', 'feature 2', 'f2', 'pageid', 'page_id', 'page id'],
        'feature8': ['feature8', 'feature_8', 'feature 8', 'f8', 'comprehension', 'comprehensionrate', 'comprehension_rate', 'correct_rate', 'correctrate'],
        # optional raw features used for Education/Gender/ID
        'feature13': ['feature13', 'feature_13', 'feature 13', 'f13', 'education'],
        'feature14': ['feature14', 'feature_14', 'feature 14', 'f14', 'gender', 'sex'],
        'feature1': ['feature1', 'feature_1', 'feature 1', 'f1', 'id', 'recordid', 'record_id', 'uuid']
    }

    col_map: Dict[str, str] = {}
    missing_features: List[str] = []
    for feat, candidates in expected_features.items():
        found = _find_column(df, candidates)
        if found:
            col_map[feat] = found
        else:
            # only strictly require the core ones below; collect missing but continue
            missing_features.append(feat)

    # Check that the absolutely required raw features for computing the DV and main IV are present.
    required_for_computation = ['feature3', 'feature5', 'feature7']
    missing_required = [f for f in required_for_computation if f not in col_map]
    if missing_required:
        # If we can't find the core raw features and also the final model columns aren't present, raise a clear error.
        available = list(df.columns)
        raise KeyError(
            f"Input dataframe is missing required raw features for transformation: {missing_required}. "
            f"Available columns: {available}"
        )

    # Now compute/construct the final columns using mapped raw columns.
    # Reading time (ms) and words
    # Ensure ReadingTime_ms is numeric. If the matched column is 'speed' or similar that is already seconds/words, we still coerce here.
    df['ReadingTime_ms'] = pd.to_numeric(df[col_map['feature5']], errors='coerce')

    # If reading time appears to be in seconds rather than milliseconds (e.g., very small values), try to detect and convert:
    # However, do not modify original semantics: we expect milliseconds. If values look like seconds (max < 100), convert to ms.
    try:
        max_rt = df['ReadingTime_ms'].max(skipna=True)
        if pd.notna(max_rt) and max_rt <= 100:  # likely in seconds, convert to milliseconds
            df['ReadingTime_ms'] = df['ReadingTime_ms'] * 1000.0
    except Exception:
        pass

    df['Words'] = pd.to_numeric(df[col_map['feature7']], errors='coerce')

    # Drop rows with invalid or missing reading time or words
    df = df.dropna(subset=['ReadingTime_ms', 'Words'])
    df = df[(df['ReadingTime_ms'] > 0) & (df['Words'] > 0)]

    # Compute seconds per word and log transform (dependent variable)
    df['TimePerWord_s'] = (df['ReadingTime_ms'] / 1000.0) / df['Words']
    df = df[df['TimePerWord_s'] > 0]
    df['LogTimePerWord'] = np.log(df['TimePerWord_s'])

    # Independent variable: ReaderView
    df['ReaderView'] = pd.to_numeric(df[col_map['feature3']], errors='coerce').fillna(0).astype(int)

    # Control: Dyslexia (binary) and severity
    if 'feature17' in col_map:
        df['Dyslexia'] = pd.to_numeric(df[col_map['feature17']], errors='coerce').fillna(0).astype(int)
    else:
        df['Dyslexia'] = 0

    # feature12: map severity to categorical strings
    def _map_severity(x):
        try:
            xi = int(float(x))
        except Exception:
            return 'NoDyslexia'
        if xi == 0:
            return 'NoDyslexia'
        elif xi == 1:
            return 'Dyslexia'
        elif xi == 2:
            return 'SevereDyslexia'
        else:
            return 'NoDyslexia'

    if 'feature12' in col_map:
        df['DyslexiaSeverity'] = df[col_map['feature12']].apply(_map_severity).astype('category')
    else:
        df['DyslexiaSeverity'] = 'NoDyslexia'
        df['DyslexiaSeverity'] = df['DyslexiaSeverity'].astype('category')

    # Age
    if 'feature10' in col_map:
        df['Age'] = pd.to_numeric(df[col_map['feature10']], errors='coerce')
    else:
        df['Age'] = np.nan

    # Device as categorical string
    if 'feature11' in col_map:
        df['Device'] = df[col_map['feature11']].astype(str)
    else:
        df['Device'] = 'unknown'

    # Education categorical
    if 'feature13' in col_map:
        df['Education'] = df[col_map['feature13']].astype(str)
    else:
        df['Education'] = 'unknown'

    # Gender mapping
    def _map_gender(x):
        try:
            xi = int(float(x))
        except Exception:
            sx = str(x)
            # normalize common string values
            s = sx.strip().lower()
            if s in ('m', 'male'):
                return 'Male'
            if s in ('f', 'female'):
                return 'Female'
            if s in ('other', 'o', ''):
                return 'Other'
            return sx
        if xi == 0:
            return 'Male'
        elif xi == 1:
            return 'Female'
        elif xi == 2:
            return 'Other'
        else:
            return str(x)

    if 'feature14' in col_map:
        df['Gender'] = df[col_map['feature14']].apply(_map_gender).astype('category')
    else:
        df['Gender'] = 'Other'
        df['Gender'] = df['Gender'].astype('category')

    # Native English
    if 'feature18' in col_map:
        s = df[col_map['feature18']]
        s_num = pd.to_numeric(s, errors='coerce')
        if s_num.notna().any():
            # treat nonzero as 1, zero as 0
            df['NativeEnglish'] = s_num.fillna(0).astype(int).clip(lower=0, upper=1)
        else:
            # try Y/N mapping
            mapped = s.astype(str).str.strip().str.upper().map({'Y': 1, 'N': 0, 'YES': 1, 'NO': 0})
            df['NativeEnglish'] = mapped.fillna(0).astype(int)
    else:
        df['NativeEnglish'] = 0

    # Retake
    if 'feature16' in col_map:
        df['Retake'] = pd.to_numeric(df[col_map['feature16']], errors='coerce').fillna(0).astype(int)
    else:
        df['Retake'] = 0

    # Readability numeric
    if 'feature19' in col_map:
        df['Readability'] = pd.to_numeric(df[col_map['feature19']], errors='coerce')
    else:
        df['Readability'] = np.nan

    # Page ID categorical
    if 'feature2' in col_map:
        df['PageID'] = df[col_map['feature2']].astype(str)
    else:
        df['PageID'] = df.index.astype(str)

    # Comprehension rate (0-1)
    if 'feature8' in col_map:
        df['ComprehensionRate'] = pd.to_numeric(df[col_map['feature8']], errors='coerce')
    else:
        df['ComprehensionRate'] = np.nan

    # Keep only rows with non-missing values for the model inputs (conservative)
    df = df.dropna(subset=['LogTimePerWord', 'ReaderView', 'Dyslexia'])

    # Cast types to match contract
    df['ReaderView'] = df['ReaderView'].astype(int)
    df['Dyslexia'] = df['Dyslexia'].astype(int)
    df['NativeEnglish'] = df['NativeEnglish'].astype(int)
    df['Retake'] = df['Retake'].astype(int)

    # Build final keep list preserving required final column names
    keep_cols = [c for c in (model_cols + aux_cols) if c in df.columns]

    return df[keep_cols].copy()


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of log(seconds per word) on ReaderView and its interaction with Dyslexia,
    controlling for demographic and trial-level covariates. Returns the fitted model results.

    Model specification:
      LogTimePerWord ~ ReaderView * Dyslexia + Age + C(Device) + C(Education) + C(Gender)
                       + NativeEnglish + Retake + Readability + C(PageID) + ComprehensionRate

    We use heteroskedasticity-robust standard errors (HC3).
    """
    import statsmodels.formula.api as smf

    formula = (
        'LogTimePerWord ~ ReaderView * Dyslexia + Age + '
        'C(Device) + C(Education) + C(Gender) + NativeEnglish + Retake + Readability + '
        'C(PageID) + ComprehensionRate'
    )

    model_fit = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    return model_fit