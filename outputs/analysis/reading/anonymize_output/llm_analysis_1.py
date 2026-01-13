from typing import Any, List
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling. The function will:
    - Rename columns to meaningful names used in the model (if obvious raw names exist)
    - Attempt to detect common alternative column names and map them to the required final columns
    - Compute ReadingSpeedWPM using the number of words and reading time excluding scrolling
    - Create cleaned/typed versions of key variables (ReaderView, Dyslexia, NativeEnglish, etc.)
    - Drop rows with invalid or missing values for variables required by the model
    - Return the transformed dataframe containing all columns referenced by the model
    """
    df = df.copy()

    # Preferred explicit renames for known raw feature names
    rename_map = {
        'feature1': 'ParticipantID',
        'feature3': 'ReaderView',          # 1 = reader view on, 0 = off
        'feature4': 'TimeOnPageMS',        # total time on page (ms)
        'feature5': 'ReadingTimeMS',       # time on page minus scrolling (ms)
        'feature6': 'ScrollTimeMS',        # scrolling time (ms)
        'feature7': 'Words',               # number of words on page
        'feature8': 'ComprehensionRate',
        'feature10': 'Age',
        'feature11': 'Device',
        'feature12': 'DyslexiaSeverity',
        'feature13': 'Education',
        'feature14': 'Gender',
        'feature15': 'Language',
        'feature16': 'Retake',             # 1 = retake, 0 = not
        'feature17': 'Dyslexia',           # binary dyslexia flag (1 = dyslexia, 0 = no)
        'feature18': 'NativeEnglish',      # 'Y'/'N'
        'feature19': 'FleschKincaid'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Helper to map likely alternative incoming column names to required final names
    def map_if_missing(target: str, name_fragments: List[str]):
        if target in df.columns:
            return
        lower_cols = {col: col.lower().replace(' ', '').replace('_', '') for col in df.columns}
        for col, simplified in lower_cols.items():
            for frag in name_fragments:
                if frag in simplified:
                    df[target] = df[col]
                    return

    # Map a number of expected conceptual variables if they exist under slightly different names
    map_if_missing('ReaderView', ['readerview', 'readerviewon', 'readerviewoff', 'readerviewflag', 'readerviewmode', 'readermode', 'reader', 'rv'])
    map_if_missing('ParticipantID', ['participantid', 'participant', 'subject', 'subj', 'id'])
    map_if_missing('Dyslexia', ['dyslexia', 'dyslexic'])
    map_if_missing('NativeEnglish', ['nativeenglish', 'native', 'englishnative', 'englishfirst'])
    map_if_missing('Words', ['words', 'wordcount', 'word_count', 'nwords', 'n_words'])
    map_if_missing('ReadingTimeMS', ['readingtimems', 'reading_time_ms', 'readingtimemillis', 'readingms', 'readingtime_ms'])
    map_if_missing('TimeOnPageMS', ['timeonpagems', 'timeonpage', 'time_on_page', 'timeonpage_ms'])
    map_if_missing('ScrollTimeMS', ['scrolltimems', 'scrolltime', 'scroll_time', 'scroll_time_ms'])
    map_if_missing('FleschKincaid', ['fleschkincaid', 'flesch', 'fk_score', 'fk'])
    map_if_missing('Retake', ['retake', 'retest'])
    map_if_missing('Age', ['age', 'years'])
    map_if_missing('Device', ['device', 'platform'])
    map_if_missing('Education', ['education', 'edu'])
    map_if_missing('Gender', ['gender', 'sex'])

    # Helper to create ReadingTimeMS when missing:
    # 1) If ReadingTimeMS exists, keep it.
    # 2) Else if TimeOnPageMS and ScrollTimeMS exist, compute ReadingTimeMS = TimeOnPageMS - ScrollTimeMS.
    # 3) Else if a column that looks like reading time in seconds exists, use it * 1000.
    if 'ReadingTimeMS' not in df.columns:
        # Coerce TimeOnPageMS/ScrollTimeMS if present
        if ('TimeOnPageMS' in df.columns) and ('ScrollTimeMS' in df.columns):
            df['TimeOnPageMS'] = pd.to_numeric(df['TimeOnPageMS'], errors='coerce')
            df['ScrollTimeMS'] = pd.to_numeric(df['ScrollTimeMS'], errors='coerce')
            df['ReadingTimeMS'] = df['TimeOnPageMS'] - df['ScrollTimeMS']
        else:
            # Option B: look for a seconds-based reading time column and convert to ms
            sec_candidates = ['readingtimeseconds', 'readingtimesec', 'readingtimes', 'readingtimesecond', 'reading_time_s', 'reading_seconds', 'readingsecs']
            found = False
            for col in list(df.columns):
                simplified = col.lower().replace(' ', '').replace('_', '')
                if simplified in sec_candidates or any(s in simplified for s in ['readingtimesecond', 'readingsecs', 'readingseconds', 'readingtimesec']):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df['ReadingTimeMS'] = df[col] * 1000.0
                    found = True
                    break
            if not found:
                # As a final fallback, try to coerce any column named similarly with different casing
                for existing in list(df.columns):
                    if existing.lower().replace(' ', '').replace('_', '') in ('readingtimems', 'readingtimemilliseconds', 'reading_time_ms'):
                        df['ReadingTimeMS'] = pd.to_numeric(df[existing], errors='coerce')
                        found = True
                        break

    # Ensure numeric types where expected
    numeric_cols = ['ReadingTimeMS', 'TimeOnPageMS', 'ScrollTimeMS', 'Words', 'Age', 'FleschKincaid', 'Retake', 'Dyslexia', 'DyslexiaSeverity', 'Gender']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # ReaderView should be 0/1 integer where possible
    if 'ReaderView' in df.columns:
        rv = df['ReaderView']
        # Try to coerce directly to numeric first
        rv_numeric = pd.to_numeric(rv, errors='coerce')
        # For non-numeric entries, attempt mapping from common string encodings
        mask_non_numeric = rv_numeric.isna()
        if mask_non_numeric.any():
            rv_mapped = rv.astype(str).str.strip().str.upper().map({
                'Y': 1, 'YES': 1, 'TRUE': 1, '1': 1, 'ON': 1,
                'N': 0, 'NO': 0, 'FALSE': 0, '0': 0, 'OFF': 0
            })
            rv_mapped = pd.to_numeric(rv_mapped, errors='coerce')
            still_na = mask_non_numeric & rv_mapped.isna()
            if still_na.any():
                parsed = pd.to_numeric(rv.astype(str).str.extract(r'(\d+)')[0], errors='coerce')
                # Align indices before assignment
                rv_mapped.loc[still_na] = parsed.loc[still_na]
            combined = rv_numeric.copy()
            combined[combined.isna()] = rv_mapped[combined.isna()]
            df['ReaderView'] = combined
        else:
            df['ReaderView'] = rv_numeric

    # Dyslexia binary indicator: coerce to numeric but also map common string encodings
    if 'Dyslexia' in df.columns:
        s = df['Dyslexia']
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            df['Dyslexia'] = pd.to_numeric(s, errors='coerce')
        else:
            mapped = s.astype(str).str.strip().str.upper().map({
                'Y': 1, 'YES': 1, 'TRUE': 1, '1': 1, 'DYSLEXIC': 1, 'DYSLEXIA': 1, 'DX': 1,
                'N': 0, 'NO': 0, 'FALSE': 0, '0': 0, 'NOTDYSLEXIC': 0, 'NONE': 0
            })
            df['Dyslexia'] = pd.to_numeric(mapped, errors='coerce')
        # If Dyslexia currently contains severity-like numeric values (e.g., 0/1/2/3),
        # convert non-missing numeric values to binary indicator: >0 -> 1, ==0 -> 0.
        if df['Dyslexia'].notna().any():
            non_na_idx = df['Dyslexia'].notna()
            df.loc[non_na_idx, 'Dyslexia'] = (df.loc[non_na_idx, 'Dyslexia'] > 0).astype(int)

    # Native English: map 'Y'/'N' or 'Yes'/'No' or numeric to 1/0
    if 'NativeEnglish' in df.columns:
        s = df['NativeEnglish']
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            df['NativeEnglish'] = pd.to_numeric(s, errors='coerce')
        else:
            mapped = s.astype(str).str.strip().str.upper().map({
                'Y': 1, 'YES': 1, 'TRUE': 1, '1': 1,
                'N': 0, 'NO': 0, 'FALSE': 0, '0': 0
            })
            df['NativeEnglish'] = pd.to_numeric(mapped, errors='coerce')

    # Compute reading speed (WPM). Use ReadingTimeMS (ms) and Words.
    # ReadingSpeedWPM = Words * 60000 / ReadingTimeMS
    df['ReadingSpeedWPM'] = np.nan

    # Ensure Words numeric
    if 'Words' in df.columns:
        df['Words'] = pd.to_numeric(df['Words'], errors='coerce')

    # Prepare reading_time_used_ms with fallbacks:
    # Prefer ReadingTimeMS when >0; else use TimeOnPageMS when >0.
    reading_time_used = pd.Series(index=df.index, dtype='float64')

    if 'ReadingTimeMS' in df.columns:
        reading_time_used = df['ReadingTimeMS'].astype('float64').copy()
    else:
        reading_time_used = pd.Series(np.nan, index=df.index)

    # For rows where reading_time_used is missing or non-positive, try TimeOnPageMS
    if 'TimeOnPageMS' in df.columns:
        time_on_page = df['TimeOnPageMS'].astype('float64')
        mask_use_timeon = (reading_time_used.isna()) | (reading_time_used <= 0)
        if mask_use_timeon.any():
            reading_time_used.loc[mask_use_timeon] = time_on_page.loc[mask_use_timeon]

    # Final compute where possible
    mask_valid_time = reading_time_used.notna() & (reading_time_used > 0)
    mask_valid_words = ('Words' in df.columns) & df['Words'].notna() & (df['Words'] > 0)
    mask_compute = mask_valid_time & mask_valid_words
    if mask_compute.any():
        df.loc[mask_compute, 'ReadingSpeedWPM'] = (df.loc[mask_compute, 'Words'] * 60000.0) / reading_time_used.loc[mask_compute]

    # If no rows could be computed using ReadingTimeMS or TimeOnPageMS, try a looser fallback:
    # If any rows still have missing ReadingSpeedWPM but Words>0 and TimeOnPageMS is present (and positive),
    # attempt to use TimeOnPageMS as an approximation (only as a last resort). Warn the user.
    if df['ReadingSpeedWPM'].isna().all():
        fallback_used = False
        if 'TimeOnPageMS' in df.columns and 'Words' in df.columns:
            mask_fallback = df['TimeOnPageMS'].notna() & (df['TimeOnPageMS'] > 0) & df['Words'].notna() & (df['Words'] > 0)
            if mask_fallback.any():
                df.loc[mask_fallback, 'ReadingSpeedWPM'] = (df.loc[mask_fallback, 'Words'] * 60000.0) / df.loc[mask_fallback, 'TimeOnPageMS']
                fallback_used = True
                warnings.warn("ReadingSpeedWPM computed using TimeOnPageMS as a fallback where ReadingTimeMS was unavailable.", UserWarning)
        if (not fallback_used) and ('ScrollTimeMS' in df.columns) and ('Words' in df.columns):
            mask_fallback2 = df['ScrollTimeMS'].notna() & (df['ScrollTimeMS'] > 0) & df['Words'].notna() & (df['Words'] > 0)
            if mask_fallback2.any():
                df.loc[mask_fallback2, 'ReadingSpeedWPM'] = (df.loc[mask_fallback2, 'Words'] * 60000.0) / df.loc[mask_fallback2, 'ScrollTimeMS']
                warnings.warn("ReadingSpeedWPM computed using ScrollTimeMS as a last-resort fallback.", UserWarning)

    # Basic cleaning: drop rows that are missing the DV or the primary IV or the dyslexia moderator
    required_cols = ['ReadingSpeedWPM', 'ReaderView', 'Dyslexia', 'ParticipantID']
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in input dataframe after renaming and preprocessing.")

    # Keep rows where the required variables are present
    df = df[df['ReadingSpeedWPM'].notna()]
    df = df[df['ReaderView'].notna()]
    df = df[df['Dyslexia'].notna()]

    # Drop impossible / extreme values: remove non-positive WPM
    df = df[df['ReadingSpeedWPM'] > 0]

    # Cast categorical/tidy columns so downstream modeling can treat them easily
    if 'Device' in df.columns:
        # Keep as category but ensure missing values are represented as NaN
        df['Device'] = df['Device'].astype('category')
    if 'Education' in df.columns:
        df['Education'] = df['Education'].astype('category')
    if 'Gender' in df.columns:
        # Keep numeric codes, but ensure numeric dtype (float if missing values exist)
        df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce')
    if 'ParticipantID' in df.columns:
        # Ensure ParticipantID is string; preserve missingness for now (will be handled in model)
        # Do not convert NaNs to the string 'nan' — keep them as actual NaN so model can treat them robustly.
        # Achieve this by converting non-null values to string, leaving NaNs untouched.
        df.loc[df['ParticipantID'].notna(), 'ParticipantID'] = df.loc[df['ParticipantID'].notna(), 'ParticipantID'].astype(str)

    # Ensure Retake is binary 0/1
    if 'Retake' in df.columns:
        # Some datasets may encode missing retake as NaN; default missing to 0 (not a retake)
        df['Retake'] = pd.to_numeric(df['Retake'], errors='coerce').fillna(0).astype(int)

    # At this point, ensure integer-like columns that will be used in modeling are standard numpy dtypes
    # Patsy/statsmodels don't handle pandas nullable dtypes well, so convert where there are no missing values.
    int_like_cols = ['ReaderView', 'Dyslexia', 'NativeEnglish', 'Retake']
    for col in int_like_cols:
        if col in df.columns:
            if df[col].isna().any():
                # leave as float (with NaNs) to avoid casting errors; model will handle or user can filter further
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # safe to cast to numpy integer dtype
                df[col] = df[col].astype(np.int64)

    # Keep only columns that are used in the model plus a few diagnostics / helpers
    keep_cols = [
        'ParticipantID', 'ReaderView', 'Dyslexia', 'ReadingSpeedWPM', 'Age', 'Device', 'Education',
        'NativeEnglish', 'FleschKincaid', 'Retake', 'Gender', 'Words', 'ReadingTimeMS', 'ScrollTimeMS', 'DyslexiaSeverity'
    ]
    # Intersect with existing columns (in case some are absent)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Final reset index
    df = df.reset_index(drop=True)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a statistical model to test whether Reader View improves reading speed for dyslexic readers.

    Strategy:
    - Use a linear mixed-effects model (random intercept for ParticipantID) to account for repeated measures.
    - Fixed effects: ReaderView, Dyslexia, ReaderView:Dyslexia interaction (tests whether the Reader View effect differs for dyslexic readers).
    - Covariates: Age, Device (categorical), Education (categorical), NativeEnglish, FleschKincaid, Retake, Gender.

    Returns the fitted model result object, or None if no rows are available to fit.
    """
    # Verify required columns exist
    required = ['ReadingSpeedWPM', 'ReaderView', 'Dyslexia', 'ParticipantID']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' missing from dataframe passed to model().")

    df = df.copy()

    # Handle missing ParticipantID values:
    # - If some ParticipantID values are missing, replace those missing entries with unique synthetic IDs
    #   so that groups passed to the mixed model are not NA. This preserves the ParticipantID column name
    #   while allowing the model to run. These synthetic IDs are internal helpers and do not change the
    #   conceptual variable contract.
    # - Detect common textual placeholders that represent missing values ('nan', 'none', empty string)
    pid = df['ParticipantID']
    # Create mask of missing-like values
    mask_missing = pid.isna()
    # If dtype is object/str, consider string forms 'nan', 'none', '' as missing too
    if pid.dtype == object or pd.api.types.is_string_dtype(pid):
        str_vals = pid.astype(str).str.strip().str.lower()
        mask_missing = mask_missing | str_vals.isin(['nan', 'none', ''])
    # Replace missing-like ParticipantIDs with unique synthetic identifiers
    if mask_missing.any():
        missing_idxs = df.index[mask_missing]
        for i, idx in enumerate(missing_idxs):
            df.at[idx, 'ParticipantID'] = f"_anon_{i}"
    # Ensure ParticipantID is string for grouping
    df['ParticipantID'] = df['ParticipantID'].astype(str)

    if df.shape[0] == 0:
        warnings.warn("No rows remain in the dataframe passed to model(); returning None instead of fitting.", UserWarning)
        return None

    # Ensure types are compatible with patsy/statsmodels: avoid pandas nullable integer dtypes
    # Convert integer-like columns to numpy integer dtype if there are no missing values
    for col in ['ReaderView', 'Dyslexia', 'NativeEnglish', 'Retake']:
        if col in df.columns:
            if df[col].isna().any():
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(np.int64)

    # Build formula. Use C(Device) and C(Education) to treat them as categorical factors.
    formula_terms: List[str] = [
        'ReaderView',
        'Dyslexia',
        'ReaderView:Dyslexia',
        'Age',
        'NativeEnglish',
        'FleschKincaid',
        'Retake',
        'Gender'
    ]
    # Add categorical factors to the formula only if present and have at least two non-missing unique levels
    if 'Device' in df.columns:
        try:
            n_device_levels = int(df['Device'].dropna().nunique())
        except Exception:
            n_device_levels = 0
        if n_device_levels >= 2:
            formula_terms.append('C(Device)')

    if 'Education' in df.columns:
        try:
            n_edu_levels = int(df['Education'].dropna().nunique())
        except Exception:
            n_edu_levels = 0
        if n_edu_levels >= 2:
            formula_terms.append('C(Education)')

    formula = 'ReadingSpeedWPM ~ ' + ' + '.join(formula_terms)

    # Fit mixed-effects model with participant random intercept
    # Using REML=False (ML) for easier comparison if needed
    md = smf.mixedlm(formula, df, groups=df['ParticipantID'])
    mdf = md.fit(reml=False)

    # Return the fitted model object. The caller can inspect summary(), params, pvalues, etc.
    return mdf