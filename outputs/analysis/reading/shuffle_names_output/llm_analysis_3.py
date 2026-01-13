from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe containing the exact columns used in the model.

    Output columns included (and used later in the model):
      - reader_view_on: binary (1 = Reader View active, 0 = not active)
      - reading_speed_wpm: words per minute on the trial
      - is_dyslexic: binary (1 = dyslexia present (dyslexia_bin >=1), 0 = no dyslexia)
      - age, education, device, Flesch_Kincaid, num_words_on_page, scrolling_time_ms, retake_trial, page_id
    """
    df = df.copy()

    # ---------- Create reader_view_on (treatment indicator) ----------
    # Primary source: 'running_time' is documented as indicator (1 = reader view activated, 0 = not)
    if 'running_time' in df.columns:
        # Ensure numeric and binary
        try:
            df['reader_view_on'] = pd.to_numeric(df['running_time'], errors='coerce').fillna(0).astype(int).apply(lambda x: 1 if x == 1 else 0)
        except Exception:
            df['reader_view_on'] = df['running_time'].map({1: 1, '1': 1, 0: 0, '0': 0}).fillna(0).astype(int)
    elif 'reader_view' in df.columns:
        # Alternate encoding: 'Y'/'N'
        df['reader_view_on'] = df['reader_view'].map({'Y': 1, 'y': 1, 'N': 0, 'n': 0}).fillna(0).astype(int)
    else:
        # If neither exists, create an all-NA column so downstream filtering will remove rows
        df['reader_view_on'] = np.nan

    # ---------- Construct reading time in milliseconds ----------
    # Prefer 'language' (documented as time on page minus scrolling), else compute adjusted_running_time - scrolling_time
    if 'language' in df.columns:
        df['reading_time_ms'] = pd.to_numeric(df['language'], errors='coerce')
    else:
        df['reading_time_ms'] = np.nan

    # fallback: adjusted_running_time - scrolling_time
    if df['reading_time_ms'].isna().any() and 'adjusted_running_time' in df.columns and 'scrolling_time' in df.columns:
        adjusted = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
        scrolling = pd.to_numeric(df['scrolling_time'], errors='coerce')
        fallback = adjusted - scrolling
        # where reading_time_ms is NA use fallback
        df.loc[df['reading_time_ms'].isna(), 'reading_time_ms'] = fallback.loc[df['reading_time_ms'].isna()]

    # store scrolling_time for control
    if 'scrolling_time' in df.columns:
        df['scrolling_time_ms'] = pd.to_numeric(df['scrolling_time'], errors='coerce')
    else:
        df['scrolling_time_ms'] = np.nan

    # ---------- Determine number of words on page ----------
    num_words_col = None
    if 'dyslexia' in df.columns:
        # heuristic: if mean is > 20, treat as word count
        dv = pd.to_numeric(df['dyslexia'], errors='coerce')
        if dv.dropna().shape[0] > 0 and dv.dropna().mean() > 20:
            df['num_words_on_page'] = dv
            num_words_col = 'dyslexia'
    if num_words_col is None and 'num_words' in df.columns:
        nv = pd.to_numeric(df['num_words'], errors='coerce')
        if nv.dropna().shape[0] > 0 and nv.dropna().mean() > 20:
            df['num_words_on_page'] = nv
            num_words_col = 'num_words'
    if num_words_col is None:
        # fallback: try to coerce 'dyslexia' (even if small) or set NA
        df['num_words_on_page'] = pd.to_numeric(df.get('dyslexia', np.nan), errors='coerce')

    # ---------- Dyslexia status: create is_dyslexic (moderator) ----------
    if 'dyslexia_bin' in df.columns:
        df['is_dyslexic'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').apply(lambda x: 1 if pd.notnull(x) and x >= 1 else 0)
    else:
        # fallback: set NA so rows lacking this information will be dropped
        df['is_dyslexic'] = np.nan

    # ---------- Convert other controls to numeric where appropriate ----------
    for col in ['age', 'education', 'device', 'Flesch_Kincaid']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # keep retake_trial and page_id as-is (categorical / id)
    if 'retake_trial' in df.columns:
        try:
            df['retake_trial'] = df['retake_trial'].astype('category')
        except Exception:
            df['retake_trial'] = pd.Categorical(df['retake_trial'])
    else:
        df['retake_trial'] = pd.Categorical([np.nan] * len(df))

    if 'page_id' not in df.columns:
        df['page_id'] = np.nan

    # ---------- Compute reading speed (WPM) ----------
    df['reading_time_s'] = pd.to_numeric(df['reading_time_ms'], errors='coerce') / 1000.0
    df['num_words_on_page'] = pd.to_numeric(df['num_words_on_page'], errors='coerce')

    df['reading_speed_wpm'] = np.nan
    valid_mask = (df['num_words_on_page'].notna()) & (df['reading_time_s'].notna()) & (df['reading_time_s'] > 0.1)
    df.loc[valid_mask, 'reading_speed_wpm'] = df.loc[valid_mask, 'num_words_on_page'] * 60.0 / df.loc[valid_mask, 'reading_time_s']

    # ---------- Final filtering: keep rows with essential variables ----------
    required = ['reader_view_on', 'reading_speed_wpm', 'is_dyslexic']
    df_final = df.dropna(subset=required).copy()

    # Remove extreme or implausible speeds (optional cleaning): remove WPM > 1000 or < 10 as likely measurement errors
    df_final = df_final[(df_final['reading_speed_wpm'] > 5) & (df_final['reading_speed_wpm'] < 2000)]

    # Ensure integer / dtype consistency for modeling
    if 'reader_view_on' in df_final.columns:
        # safe cast: ensure values are 0/1
        df_final['reader_view_on'] = pd.to_numeric(df_final['reader_view_on'], errors='coerce').fillna(0).astype(int)
    if 'is_dyslexic' in df_final.columns:
        df_final['is_dyslexic'] = pd.to_numeric(df_final['is_dyslexic'], errors='coerce').fillna(0).astype(int)

    # Keep only the columns we declared in the conceptual model (plus reading_time_s and reading_time_ms for diagnostics)
    keep_cols = [
        'reader_view_on', 'reading_speed_wpm', 'is_dyslexic', 'age', 'education', 'device', 'Flesch_Kincaid',
        'num_words_on_page', 'scrolling_time_ms', 'retake_trial', 'page_id', 'reading_time_ms', 'reading_time_s'
    ]
    # If any of these columns are missing just add them as NA to keep a consistent schema
    for col in keep_cols:
        if col not in df_final.columns:
            df_final[col] = np.nan

    # Ensure consistent column ordering
    df_final = df_final[keep_cols]

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model testing whether Reader View improves reading speed specifically for participants with dyslexia.

    Primary specification:
      reading_speed_wpm ~ reader_view_on * is_dyslexic + age + education + device + Flesch_Kincaid + num_words_on_page + scrolling_time_ms

    Returns the fitted statsmodels regression results object (with robust HC3 standard errors printed).
    If there are no usable observations after cleaning, returns None.
    """
    # Work on a copy
    df = df.copy()

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Ensure all predictor columns exist in the dataframe (add as NA if missing) to make dropna predictable
    predictors = [
        'reader_view_on', 'is_dyslexic', 'age', 'education', 'device',
        'Flesch_Kincaid', 'num_words_on_page', 'scrolling_time_ms'
    ]
    for col in predictors + ['reading_speed_wpm']:
        if col not in df.columns:
            df[col] = np.nan

    # Convert predictors to numeric where appropriate to avoid object dtype issues
    for col in ['reader_view_on', 'is_dyslexic', 'age', 'education', 'device', 'Flesch_Kincaid', 'num_words_on_page', 'scrolling_time_ms', 'reading_speed_wpm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing any variable used in the formula (including the DV).
    required_for_model = ['reading_speed_wpm'] + predictors
    df_model = df.dropna(subset=required_for_model).copy()

    # If no data remains after dropping missing required vars, do not attempt to fit the model
    if df_model.shape[0] == 0:
        print("No observations available after dropping missing values for required variables. Returning None.")
        return None

    # Build formula with interaction between reader_view_on and is_dyslexic
    formula = (
        'reading_speed_wpm ~ reader_view_on * is_dyslexic '
        '+ age + education + device + Flesch_Kincaid + num_words_on_page + scrolling_time_ms'
    )

    # Fit OLS with robust standard errors (HC3)
    try:
        model_res = smf.ols(formula=formula, data=df_model).fit(cov_type='HC3')
    except Exception as e:
        # If fitting fails for any reason, report and return None
        print(f"Model fitting failed: {e}")
        return None

    # Print summary for quick inspection
    print(model_res.summary())

    # Return the fitted results object for downstream inspection
    return model_res