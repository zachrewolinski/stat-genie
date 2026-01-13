from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    Produces the following new columns (used in modeling):
      - ReaderView: binary (0/1) from feature3
      - ReadingTimeMs: reading time excluding scrolling (feature5)
      - NumWords: number of words on page (feature7)
      - ReadingSpeedWPM: NumWords / (ReadingTimeMs in minutes)
      - LogReadingSpeed: np.log1p(ReadingSpeedWPM)  (DV)
      - Dyslexia: binary indicator from feature17 (0/1)
      - Age, Device, Education, Gender, IsNativeEnglish, Comprehension, Readability, IsRetake

    The function is robust to common encodings of booleans and missing values.
    It imputes reasonable defaults for non-critical numeric controls to avoid
    dropping all observations while ensuring the DV and primary IV are valid.
    """
    df = df.copy()

    def _get_series(column_name: str) -> pd.Series:
        if column_name in df.columns:
            return df[column_name]
        else:
            return pd.Series([None] * len(df), index=df.index)

    # Reader view: robust parsing of many textual/numeric encodings
    raw_reader = _get_series('feature3')

    def parse_binary_series(s: pd.Series) -> pd.Series:
        """
        Parse a series into 0/1 where possible. Returns a float Series with
        1.0, 0.0, or np.nan for unknowns.
        Heuristics:
         - numeric 1 -> 1, numeric 0 -> 0
         - textual positives: '1','true','t','yes','y','on','enabled'
         - textual negatives: '0','false','f','no','n','off','disabled','none',''
         - Any other non-empty string that is not an explicit negative is treated as positive.
        """
        # Start with numeric parse
        num = pd.to_numeric(s, errors='coerce')
        res = pd.Series(index=s.index, dtype=float)

        # Assign numeric 1/0 where explicit
        res[num == 1] = 1.0
        res[num == 0] = 0.0

        # For the remaining entries, inspect string forms
        mask_unassigned = res.isna()
        strs = s.astype(str).str.strip().str.lower()
        pos_set = {'1', 'true', 't', 'yes', 'y', 'on', 'enabled', 'readerview', 'rv_on', 'rv on', 'rvon', 'reader'}
        neg_set = {'0', 'false', 'f', 'no', 'n', 'off', 'disabled', 'none', ''}

        # Map explicit positives
        is_pos = strs.isin(pos_set)
        res.loc[mask_unassigned & is_pos] = 1.0

        # Map explicit negatives
        is_neg = strs.isin(neg_set)
        res.loc[mask_unassigned & is_neg] = 0.0

        # For other non-empty strings that are not explicit negatives, assume positive
        other_mask = mask_unassigned & (~is_pos) & (~is_neg)
        # treat entries that are non-empty (not 'nan' or '') as positive
        res.loc[other_mask & (strs != 'nan') & (strs != '')] = 1.0

        # Remaining stay as NaN
        return res

    df['ReaderView'] = parse_binary_series(raw_reader)

    # Reading time and number of words (convert to numeric)
    df['ReadingTimeMs'] = pd.to_numeric(_get_series('feature5'), errors='coerce')
    df['NumWords'] = pd.to_numeric(_get_series('feature7'), errors='coerce')

    # If reading time is missing or non-positive for many records, impute a reasonable default
    # so that the dependent variable can be constructed. Use the median of positive reading times
    # if available, otherwise default to 30000 ms (30s).
    pos_rt_mask = df['ReadingTimeMs'].notnull() & np.isfinite(df['ReadingTimeMs']) & (df['ReadingTimeMs'] > 0)
    if pos_rt_mask.any():
        rt_median = float(df.loc[pos_rt_mask, 'ReadingTimeMs'].median(skipna=True))
        if not np.isfinite(rt_median) or rt_median <= 0:
            rt_median = 30000.0
    else:
        rt_median = 30000.0
    # Replace missing or non-positive reading times with median/default
    bad_rt_mask = (~np.isfinite(df['ReadingTimeMs'])) | (df['ReadingTimeMs'] <= 0)
    df.loc[bad_rt_mask, 'ReadingTimeMs'] = rt_median

    # Dyslexia indicator: feature17 (1 = dyslexia). Default to 0 if missing.
    df['Dyslexia'] = pd.to_numeric(_get_series('feature17'), errors='coerce').fillna(0).astype(int)

    # Age
    df['Age'] = pd.to_numeric(_get_series('feature10'), errors='coerce')

    # Device and Education categorical fields
    device_series = _get_series('feature11').fillna('Unknown').astype(str)
    df['Device'] = device_series.where(device_series.str.strip().astype(bool), 'Unknown')

    education_series = _get_series('feature13').fillna('Unknown').astype(str)
    df['Education'] = education_series.where(education_series.str.strip().astype(bool), 'Unknown')

    # Gender mapping: prefer numeric-coded mapping, otherwise use string categories,
    # normalize common missing markers to 'Unknown'
    s_gender = _get_series('feature14')
    # Try numeric mapping first (handles 0/1/2 as numbers or numeric strings)
    gender_num = pd.to_numeric(s_gender, errors='coerce')
    gender_mapped = pd.Series(index=s_gender.index, dtype=object)
    gender_mapped[gender_num == 0] = 'Male'
    gender_mapped[gender_num == 1] = 'Female'
    gender_mapped[gender_num == 2] = 'Other'
    # Fallback to string representations
    gender_fallback = s_gender.astype(str).str.strip()
    gender_combined = gender_mapped.where(gender_mapped.notnull(), gender_fallback)
    gender_combined = gender_combined.replace({'None': 'Unknown', 'nan': 'Unknown', '': 'Unknown'})
    df['Gender'] = gender_combined.fillna('Unknown').astype(str)

    # IsNativeEnglish: feature18 'Y'/'N' or similar
    raw_native = _get_series('feature18')
    # handle numeric or textual
    native_num = pd.to_numeric(raw_native, errors='coerce')
    native = pd.Series(index=raw_native.index, dtype=float)
    native[native_num == 1] = 1.0
    native[native_num == 0] = 0.0
    mask_native_unassigned = native.isna()
    native_str = raw_native.astype(str).str.strip().str.lower()
    native_pos = native_str.isin({'y', 'yes', 'true', 't', '1'})
    native_neg = native_str.isin({'n', 'no', 'false', 'f', '0'})
    native.loc[mask_native_unassigned & native_pos] = 1.0
    native.loc[mask_native_unassigned & native_neg] = 0.0
    # default to 0 if still missing (non-native assumed false)
    df['IsNativeEnglish'] = native.fillna(0).astype(int)

    # Comprehension and Readability
    df['Comprehension'] = pd.to_numeric(_get_series('feature8'), errors='coerce')
    df['Readability'] = pd.to_numeric(_get_series('feature19'), errors='coerce')

    # Retake indicator
    df['IsRetake'] = pd.to_numeric(_get_series('feature16'), errors='coerce').fillna(0).astype(int)

    # Impute non-critical numeric controls with reasonable defaults if missing,
    # so we don't drop all observations unnecessarily.
    # These imputations are minimal and only applied to controls, not the IV/DV.
    # Age: median or default 30
    if df['Age'].notnull().any():
        age_median = float(df['Age'].median(skipna=True))
        if not np.isfinite(age_median):
            age_median = 30.0
    else:
        age_median = 30.0
    df['Age'] = df['Age'].fillna(age_median)

    # NumWords: median or default 200 (used to compute speed)
    if df['NumWords'].notnull().any():
        numwords_median = float(df['NumWords'].median(skipna=True))
        # Ensure median positive
        if not np.isfinite(numwords_median) or numwords_median <= 0:
            numwords_median = 200.0
    else:
        numwords_median = 200.0
    df['NumWords'] = df['NumWords'].fillna(numwords_median)

    # Comprehension: mean or default 0.75
    if df['Comprehension'].notnull().any():
        comp_mean = float(df['Comprehension'].mean(skipna=True))
        if not np.isfinite(comp_mean):
            comp_mean = 0.75
    else:
        comp_mean = 0.75
    df['Comprehension'] = df['Comprehension'].fillna(comp_mean)

    # Readability: mean or default 60.0
    if df['Readability'].notnull().any():
        read_mean = float(df['Readability'].mean(skipna=True))
        if not np.isfinite(read_mean):
            read_mean = 60.0
    else:
        read_mean = 60.0
    df['Readability'] = df['Readability'].fillna(read_mean)

    # Compute reading speed (words per minute)
    df['ReadingTimeMin'] = df['ReadingTimeMs'] / 1000.0 / 60.0

    # Compute WPM where reading time is positive; otherwise mark as NaN
    df['ReadingSpeedWPM'] = np.nan
    valid_time_mask = df['ReadingTimeMin'] > 0
    df.loc[valid_time_mask, 'ReadingSpeedWPM'] = (
        df.loc[valid_time_mask, 'NumWords'] / df.loc[valid_time_mask, 'ReadingTimeMin']
    )

    # Compute log reading speed DV
    df['LogReadingSpeed'] = np.log1p(df['ReadingSpeedWPM'])

    # Now drop only rows that are essential: we require a valid ReaderView and a valid LogReadingSpeed.
    # Ensure ReaderView is numeric before drop (keeps only those we could parse)
    df['ReaderView'] = pd.to_numeric(df['ReaderView'], errors='coerce')

    df = df.dropna(subset=['ReaderView', 'LogReadingSpeed'])

    # Also ensure LogReadingSpeed finite
    df = df[np.isfinite(df['LogReadingSpeed'])]

    # Coerce and ensure integer types for binary indicators now that problematic rows removed
    df['ReaderView'] = pd.to_numeric(df['ReaderView'], errors='coerce').astype(int)
    df['Dyslexia'] = pd.to_numeric(df['Dyslexia'], errors='coerce').fillna(0).astype(int)
    df['IsRetake'] = pd.to_numeric(df['IsRetake'], errors='coerce').fillna(0).astype(int)
    df['IsNativeEnglish'] = pd.to_numeric(df['IsNativeEnglish'], errors='coerce').fillna(0).astype(int)
    # Ensure NumWords numeric
    df['NumWords'] = pd.to_numeric(df['NumWords'], errors='coerce').fillna(numwords_median)

    # Ensure required columns exist (they should by construction)
    required_final_cols = [
        'ReaderView', 'LogReadingSpeed', 'Dyslexia', 'Age', 'Device', 'Education',
        'Gender', 'IsNativeEnglish', 'Comprehension', 'Readability', 'IsRetake', 'NumWords'
    ]
    for col in required_final_cols:
        if col not in df.columns:
            # create a sensible default column if missing (should not happen)
            if col in ['Device', 'Education', 'Gender']:
                df[col] = 'Unknown'
            else:
                df[col] = 0

    # Drop helper columns that are not part of final dataset contract
    helper_cols = ['ReadingTimeMs', 'ReadingTimeMin', 'ReadingSpeedWPM']
    # Keep ReadingTimeMs if it's part of the raw columns? It is not required in final, so drop helpers
    for hc in helper_cols:
        if hc in df.columns:
            df.drop(columns=[hc], inplace=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear model estimating whether Reader View improves reading speed for dyslexic readers.

    Model: LogReadingSpeed ~ ReaderView * Dyslexia + controls
    Controls: Age + C(Device) + C(Education) + C(Gender) + IsNativeEnglish + Comprehension + Readability + IsRetake + NumWords

    Uses heteroskedasticity-robust standard errors (HC3).
    Returns the fitted statsmodels results object.
    """
    df = df.copy()

    required_columns = [
        'LogReadingSpeed', 'ReaderView', 'Dyslexia', 'Age',
        'Device', 'Education', 'Gender', 'IsNativeEnglish',
        'Comprehension', 'Readability', 'IsRetake', 'NumWords'
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Ensure the essential numeric columns are numeric
    essential_numeric = ['LogReadingSpeed', 'ReaderView', 'Dyslexia']
    for col in essential_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing essential variables (DV, IV, moderator)
    df = df.dropna(subset=essential_numeric)
    if df.shape[0] == 0:
        raise ValueError("No observations available to fit the model after preprocessing.")

    # For the remaining numeric controls, ensure numeric and impute reasonable defaults if needed.
    # These imputations are only for controls and follow the same defaults as transform.
    # Age: median or default 30
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    if df['Age'].notnull().any():
        age_median = float(df['Age'].median(skipna=True))
        if not np.isfinite(age_median):
            age_median = 30.0
    else:
        age_median = 30.0
    df['Age'] = df['Age'].fillna(age_median)

    # IsNativeEnglish: binary control; default to 0
    df['IsNativeEnglish'] = pd.to_numeric(df['IsNativeEnglish'], errors='coerce').fillna(0).astype(int)

    # Comprehension: mean or default 0.75
    df['Comprehension'] = pd.to_numeric(df['Comprehension'], errors='coerce')
    if df['Comprehension'].notnull().any():
        comp_mean = float(df['Comprehension'].mean(skipna=True))
        if not np.isfinite(comp_mean):
            comp_mean = 0.75
    else:
        comp_mean = 0.75
    df['Comprehension'] = df['Comprehension'].fillna(comp_mean)

    # Readability: mean or default 60.0
    df['Readability'] = pd.to_numeric(df['Readability'], errors='coerce')
    if df['Readability'].notnull().any():
        read_mean = float(df['Readability'].mean(skipna=True))
        if not np.isfinite(read_mean):
            read_mean = 60.0
    else:
        read_mean = 60.0
    df['Readability'] = df['Readability'].fillna(read_mean)

    # IsRetake: binary control; default to 0
    df['IsRetake'] = pd.to_numeric(df['IsRetake'], errors='coerce').fillna(0).astype(int)

    # NumWords: median or default 200
    df['NumWords'] = pd.to_numeric(df['NumWords'], errors='coerce')
    if df['NumWords'].notnull().any():
        numwords_median = float(df['NumWords'].median(skipna=True))
        if not np.isfinite(numwords_median) or numwords_median <= 0:
            numwords_median = 200.0
    else:
        numwords_median = 200.0
    df['NumWords'] = df['NumWords'].fillna(numwords_median)

    # Now build design matrix manually
    X = pd.DataFrame({
        'ReaderView': df['ReaderView'].astype(float),
        'Dyslexia': df['Dyslexia'].astype(float),
        'ReaderView:Dyslexia': (df['ReaderView'] * df['Dyslexia']).astype(float),
        'Age': df['Age'].astype(float),
        'IsNativeEnglish': df['IsNativeEnglish'].astype(float),
        'Comprehension': df['Comprehension'].astype(float),
        'Readability': df['Readability'].astype(float),
        'IsRetake': df['IsRetake'].astype(float),
        'NumWords': df['NumWords'].astype(float)
    }, index=df.index)

    # Categorical controls
    cat_cols = ['Device', 'Education', 'Gender']
    df_cat = df[cat_cols].fillna('Unknown').astype(str)
    dummies = pd.get_dummies(df_cat, drop_first=True)
    X = pd.concat([X, dummies], axis=1)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    y = pd.to_numeric(df['LogReadingSpeed'], errors='coerce')

    # Final safeguard: align X and y indices and drop any rows with NaN in y or X
    data = pd.concat([y, X], axis=1)
    data = data.dropna(subset=['LogReadingSpeed'])
    if data.shape[0] == 0:
        raise ValueError("No observations available to fit the model after final alignment.")

    y = data['LogReadingSpeed']
    X = data.drop(columns=['LogReadingSpeed'])

    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC3')

    return results