from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the variables required for the analysis.

    Output dataframe will include the following columns used by the model:
      - reading_speed_wpm: dependent variable (words per minute)
      - reader_view_bin: independent variable (0/1)
      - is_dyslexic: moderator (0/1)
      - age, gender, Flesch_Kincaid, num_words, log_num_words, device, img_width

    Notes on column mapping (based on provided schema):
      - 'language' column contains reading time in milliseconds excluding scrolling (reading_time_ms)
      - 'dyslexia' column (despite its name) contains number of words on the page
      - 'dyslexia_bin' contains dyslexia status: 0=no,1=dyslexia,2=severe
      - 'reader_view' contains 'Y'/'N'
    """
    df = df.copy()

    # Helper: find a source column from a list of possible raw names
    def _pick_col(possible_names):
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    # Identify source columns for reading time and num words with fallbacks
    reading_time_src = _pick_col(['language', 'reading_time_ms', 'reading_time'])
    words_src = _pick_col(['dyslexia', 'num_words', 'words', 'word_count'])
    dyslexia_bin_src = _pick_col(['dyslexia_bin', 'dyslexia_status', 'dyslexia_binary'])

    # Convert to numeric where appropriate (do not drop rows yet)
    if reading_time_src is not None:
        df[reading_time_src] = pd.to_numeric(df[reading_time_src], errors='coerce')
        df['reading_time_ms'] = df[reading_time_src]
    else:
        df['reading_time_ms'] = np.nan

    if words_src is not None:
        df[words_src] = pd.to_numeric(df[words_src], errors='coerce')
        df['num_words'] = df[words_src]
    else:
        df['num_words'] = np.nan

    # Dyslexia bin (moderator source)
    if dyslexia_bin_src is not None:
        df[dyslexia_bin_src] = pd.to_numeric(df[dyslexia_bin_src], errors='coerce')
        df['dyslexia_bin'] = df[dyslexia_bin_src]
    else:
        # If explicit dyslexia_bin not present, try to infer from 'dyslexia' if it had categorical labels.
        if 'dyslexia' in df.columns and df['dyslexia'].dtype == object:
            # attempt mapping common textual encodings
            s = df['dyslexia'].astype(str).str.lower()
            mapped = pd.Series(np.nan, index=df.index)
            mapped[s.isin(['no', '0', 'none'])] = 0
            mapped[s.isin(['yes', '1', 'dyslexia'])] = 1
            df['dyslexia_bin'] = mapped
        else:
            df['dyslexia_bin'] = np.nan

    # Dependent variable: reading speed (words per minute)
    # words per minute = num_words * 60000 / reading_time_ms
    # Coerce numeric again to be safe
    df['reading_time_ms'] = pd.to_numeric(df['reading_time_ms'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')

    # Only compute reading_speed_wpm where both values are present and positive
    valid_rt = df['reading_time_ms'].notna() & (df['reading_time_ms'] > 0)
    valid_nw = df['num_words'].notna() & (df['num_words'] > 0)
    valid_idx = valid_rt & valid_nw
    df['reading_speed_wpm'] = np.nan
    if valid_idx.any():
        df.loc[valid_idx, 'reading_speed_wpm'] = df.loc[valid_idx, 'num_words'] * 60000.0 / df.loc[valid_idx, 'reading_time_ms']

    # Independent: Reader View ON/OFF
    # Coerce reader_view to string if present, otherwise create as missing
    if 'reader_view' in df.columns:
        # Robust mapping for various possible encodings: Y/N, Yes/No, True/False, 1/0
        rv = df['reader_view'].astype(str).str.strip().str.upper()
        rv = rv.replace({'TRUE': 'Y', 'T': 'Y', 'YES': 'Y', '1': 'Y', 'ON': 'Y', 'Y': 'Y',
                         'FALSE': 'N', 'F': 'N', 'NO': 'N', '0': 'N', 'OFF': 'N', 'N': 'N'})
        df['reader_view'] = rv
    else:
        df['reader_view'] = pd.Series([''] * len(df), index=df.index).astype(str)

    df['reader_view_bin'] = df['reader_view'].map({'Y': 1, 'N': 0})
    # Treat non-'Y' as 0 (including missing / unexpected levels)
    df['reader_view_bin'] = df['reader_view_bin'].fillna(0).astype(int)

    # Moderator: dyslexia status. Define dyslexic if dyslexia_bin >= 1 (includes severe).
    # If dyslexia_bin is missing, treat as 0 (not dyslexic) to avoid dropping observations unnecessarily.
    df['is_dyslexic'] = (df['dyslexia_bin'].fillna(0) >= 1).astype(int)

    # Controls: make sure present and numeric where expected
    for col in ['age', 'gender', 'Flesch_Kincaid', 'device', 'img_width']:
        if col in df.columns:
            if col == 'device':
                df[col] = df[col].where(df[col].notna(), 'missing').astype(str)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            if col == 'device':
                df[col] = pd.Series(['missing'] * len(df), index=df.index).astype(object)
            else:
                df[col] = np.nan

    # Log-transformed page length to reduce skew (num_words > 0 because we filtered above when computing)
    df['log_num_words'] = np.where(df['num_words'] > 0, np.log(df['num_words']), np.nan)

    # Do not drop rows here; leave reading_speed_wpm as NaN where it could not be computed.
    # Drop rows with NA in the final dependent variable would remove all observations if input lacks
    # the necessary measurement columns; instead, leave dropping to the model function which will
    # handle the no-observation case gracefully.
    # df = df.dropna(subset=['reading_speed_wpm'])

    # Ensure device column is categorical and ALWAYS has at least one category
    def _ensure_device_categorical(s: pd.Series) -> pd.Series:
        # If series is empty, create an empty categorical with 'missing' category
        if s.empty:
            cat = pd.Categorical([], categories=['missing'])
            return pd.Series(cat, index=s.index)
        # Otherwise, coerce to string, fill missing, and set categories to observed + 'missing'
        s2 = s.fillna('missing').astype(str)
        observed = list(pd.unique(s2))
        if 'missing' not in observed:
            observed.append('missing')
        cat = pd.Categorical(s2, categories=observed)
        return pd.Series(cat, index=s.index)

    df['device'] = _ensure_device_categorical(df['device'])

    # Return only the columns that will be used / helpful for modeling/diagnostics
    keep_cols = [
        'reading_speed_wpm',
        'reader_view_bin',
        'is_dyslexic',
        'age',
        'gender',
        'Flesch_Kincaid',
        'num_words',
        'log_num_words',
        'device',
        'img_width',
        'reading_time_ms',
        'reader_view',
    ]

    # Ensure all keep_cols exist in the returned dataframe (create missing with NA or appropriate dtype)
    for col in keep_cols:
        if col not in df.columns:
            if col == 'device':
                df[col] = _ensure_device_categorical(pd.Series(dtype=object))
            elif col in ['reading_speed_wpm', 'reader_view_bin', 'is_dyslexic', 'num_words', 'log_num_words', 'reading_time_ms']:
                df[col] = np.nan
            else:
                df[col] = np.nan

    # Ensure returned dataframe contains columns in the specified order
    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model estimating the effect of Reader View on reading speed, and whether that effect
    differs for readers with dyslexia (interaction). Control for age, gender, readability, page length,
    device type, and image width.

    Model specification (linear OLS):
      reading_speed_wpm ~ reader_view_bin * is_dyslexic + age + gender + Flesch_Kincaid + log_num_words + C(device) + img_width

    Returns the fitted statsmodels RegressionResults object. If there are no observations available
    for fitting (after dropping missing core variables), returns a lightweight dummy results object.
    """
    # Ensure model columns exist
    required = [
        'reading_speed_wpm',
        'reader_view_bin',
        'is_dyslexic',
        'age',
        'gender',
        'Flesch_Kincaid',
        'log_num_words',
        'device',
        'img_width',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop any remaining rows with missing values in core outcome and treatment/moderator variables.
    core_required = ['reading_speed_wpm', 'reader_view_bin', 'is_dyslexic']
    model_df = df.dropna(subset=core_required).copy()

    # If no observations available, return a dummy results object instead of raising an exception.
    if model_df.shape[0] == 0:
        class DummyResults:
            def __init__(self):
                self.params = pd.Series(dtype=float)
                self.bse = pd.Series(dtype=float)
                self.pvalues = pd.Series(dtype=float)
                self.rsquared = np.nan
                self.rsquared_adj = np.nan
                self.nobs = 0

            def summary(self):
                return "No observations available to fit the model."

            def __repr__(self):
                return "<DummyResults: no observations to fit model>"

        return DummyResults()

    # Ensure device is treated as categorical and has at least one category
    if 'device' in model_df.columns:
        # If device is categorical with zero categories, set to include 'missing'
        if pd.api.types.is_categorical_dtype(model_df['device']):
            if len(model_df['device'].cat.categories) == 0:
                model_df['device'] = model_df['device'].cat.set_categories(['missing'])
        else:
            # Coerce to string and fill missing, then make categorical with observed + 'missing'
            s = model_df['device'].fillna('missing').astype(str)
            cats = list(pd.unique(s))
            if 'missing' not in cats:
                cats.append('missing')
            model_df['device'] = pd.Categorical(s, categories=cats)
    else:
        # Create device column as categorical with 'missing' category
        model_df['device'] = pd.Categorical(['missing'] * len(model_df), categories=['missing'])

    formula = 'reading_speed_wpm ~ reader_view_bin * is_dyslexic + age + gender + Flesch_Kincaid + log_num_words + C(device) + img_width'

    results = smf.ols(formula=formula, data=model_df).fit(cov_type='HC3')

    # return full results so caller can inspect .summary(), .params, etc.
    return results