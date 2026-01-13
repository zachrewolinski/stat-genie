from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# NOTE: The top-level CSV read is kept as in the original submission environment.
# If this file is imported in a different environment, the path may need adjustment.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/shuffle_names_output/reading.csv')
except Exception:
    # If the file is not present, don't raise at import time; users will pass their own DataFrame to transform().
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work with a copy
    df = df.copy()

    # -----------------------
    # Create/standardize key variables
    # -----------------------
    # RunningView: prefer numeric running_time (0/1). If missing, fall back on reader_view (Y/N).
    if 'running_time' in df.columns:
        running_vals = pd.to_numeric(df['running_time'], errors='coerce')
    else:
        running_vals = pd.Series([np.nan] * len(df), index=df.index)

    if 'reader_view' in df.columns:
        # reader_view has values 'Y'/'N' in this dataset; map to 1/0
        reader_view_flag = df['reader_view'].astype(str).str.upper().map({'Y': 1, 'N': 0})
    else:
        reader_view_flag = pd.Series([np.nan] * len(df), index=df.index)

    # Start with running_vals, fill missing from reader_view_flag, then default to 0
    RunningView = running_vals.fillna(reader_view_flag).fillna(0)
    # Ensure integer 0/1
    RunningView = pd.to_numeric(RunningView, errors='coerce').fillna(0).astype(int)
    df['RunningView'] = RunningView

    # participant identifier for clustering (the dataset describes 'correct_rate' as a unique id per record)
    # Use the provided unique identifier column as participant_id; if it is missing use 'uuid' or row index
    if 'correct_rate' in df.columns:
        df['participant_id'] = df['correct_rate'].astype(str)
    elif 'uuid' in df.columns:
        df['participant_id'] = df['uuid'].astype(str)
    else:
        # fallback: use row index
        df['participant_id'] = df.index.astype(str)

    # DyslexiaIndicator: combine dyslexia_bin categories (0 = no dyslexia, 1 = dyslexia, 2 = severe dyslexia)
    if 'dyslexia_bin' in df.columns:
        dys_vals = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
        df['DyslexiaIndicator'] = dys_vals.apply(lambda x: 1 if x in [1, 2] else 0)
    elif 'dyslexia' in df.columns:
        dys_raw = pd.to_numeric(df['dyslexia'], errors='coerce')
        # If dyslexia encodes 1/0 as indicator values, use that; otherwise non-binary values get mapped
        df['DyslexiaIndicator'] = dys_raw.apply(lambda x: 1 if x == 1 else 0)
    else:
        df['DyslexiaIndicator'] = 0

    # -----------------------
    # Compute net reading time (ms) and reading speed (words per second)
    # Prefer an existing net-reading-time column 'language' if it appears to contain ms values (per schema notes)
    # Otherwise compute adjusted_running_time - scrolling_time
    df['NetReadingTimeMs'] = np.nan
    if 'language' in df.columns:
        lang_vals = pd.to_numeric(df['language'], errors='coerce')
        df.loc[lang_vals > 0, 'NetReadingTimeMs'] = lang_vals.where(lang_vals > 0)

    if 'adjusted_running_time' in df.columns and 'scrolling_time' in df.columns:
        adj = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
        scr = pd.to_numeric(df['scrolling_time'], errors='coerce')
        computed = adj - scr
        df.loc[df['NetReadingTimeMs'].isna(), 'NetReadingTimeMs'] = computed[df['NetReadingTimeMs'].isna()]

    if 'adjusted_running_time' in df.columns:
        adj = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
        df.loc[df['NetReadingTimeMs'].isna(), 'NetReadingTimeMs'] = adj[df['NetReadingTimeMs'].isna()]

    # Convert to seconds
    df['ReadingTime_s'] = pd.to_numeric(df['NetReadingTimeMs'], errors='coerce') / 1000.0

    # Number of words on the page
    if 'num_words' in df.columns:
        df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    elif 'dyslexia' in df.columns:
        # in case 'dyslexia' actually encodes num words in this messy schema
        df['num_words'] = pd.to_numeric(df['dyslexia'], errors='coerce')
    else:
        df['num_words'] = np.nan

    # Compute reading speed (words per second). If num_words is zero or missing, set speed to NaN
    df['ReadingSpeed_wps'] = np.nan
    valid_time = df['ReadingTime_s'] > 0
    valid_words = df['num_words'].notna() & (df['num_words'] > 0)
    df.loc[valid_time & valid_words, 'ReadingSpeed_wps'] = (
        df.loc[valid_time & valid_words, 'num_words'] / df.loc[valid_time & valid_words, 'ReadingTime_s']
    )

    # -----------------------
    # Create / clean control variables used in models
    # -----------------------
    # device_cat: convert device numeric to string category to be used with C() in formulas
    if 'device' in df.columns:
        df['device_cat'] = df['device'].astype(str)
    else:
        df['device_cat'] = 'unknown'

    # gender, age, education, Flesch_Kincaid, img_width, page_id: ensure numeric where applicable
    for col in ['gender', 'age', 'education', 'Flesch_Kincaid', 'img_width', 'page_id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # -----------------------
    # Filtering unrealistic / missing observations
    # -----------------------
    # Remove rows lacking the main outcome or main IV or moderator
    df = df.dropna(subset=['ReadingSpeed_wps', 'RunningView', 'DyslexiaIndicator'])

    # Remove implausible reading times: reading time < 0.2 s or > 600 s (10 minutes)
    df = df[(df['ReadingTime_s'] >= 0.2) & (df['ReadingTime_s'] <= 600)]

    # Keep only finite reading speed
    df = df[np.isfinite(df['ReadingSpeed_wps'])]

    # Final: ensure types
    df['RunningView'] = df['RunningView'].astype(int)
    df['DyslexiaIndicator'] = df['DyslexiaIndicator'].astype(int)

    # Impute missing controls so that the modeling design matrix does not drop all rows
    # For numeric controls, fill missing values with the column median where available, otherwise 0
    numeric_controls = ['gender', 'age', 'education', 'Flesch_Kincaid', 'img_width']
    for col in numeric_controls:
        if col in df.columns:
            col_median = df[col].median(skipna=True)
            if np.isnan(col_median):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(col_median)
        else:
            df[col] = 0

    # Ensure device_cat exists and is string
    if 'device_cat' not in df.columns:
        df['device_cat'] = 'unknown'
    df['device_cat'] = df['device_cat'].astype(str)

    # Ensure participant_id exists and is string
    if 'participant_id' not in df.columns:
        df['participant_id'] = df.index.astype(str)
    df['participant_id'] = df['participant_id'].astype(str)

    # Ensure num_words exists (should, since used to compute ReadingSpeed_wps)
    if 'num_words' not in df.columns:
        df['num_words'] = np.nan
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')

    # Return the transformed dataframe containing all columns required by the statistical model
    # Columns provided: RunningView, ReadingSpeed_wps, DyslexiaIndicator, device_cat, gender, age,
    # education, Flesch_Kincaid, img_width, num_words, participant_id
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model testing whether Reader View improves reading speed for individuals with dyslexia.

    Model specification:
      ReadingSpeed_wps ~ RunningView * DyslexiaIndicator + controls

    Controls: categorical device, gender, age, education, Flesch_Kincaid, img_width, num_words.

    We use cluster-robust standard errors clustered by participant_id.
    """
    # Ensure required columns exist
    required = ['ReadingSpeed_wps', 'RunningView', 'DyslexiaIndicator', 'participant_id',
                'device_cat', 'gender', 'age', 'education', 'Flesch_Kincaid', 'img_width', 'num_words']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required column(s) missing from dataframe: {missing}")

    # Work with a copy to avoid modifying caller's data
    df = df.copy()

    # Drop any rows that still lack the modeling variables (patsy/statsmodels will drop rows with any NA)
    # but provide a clear error if we end up with no rows.
    df_model = df.dropna(subset=['ReadingSpeed_wps', 'RunningView', 'DyslexiaIndicator',
                                 'device_cat', 'gender', 'age', 'education', 'Flesch_Kincaid', 'img_width', 'num_words'])
    if df_model.shape[0] == 0:
        raise ValueError("No observations available for modeling after dropping rows with missing model variables.")

    # Ensure device_cat is treated as categorical string
    df_model['device_cat'] = df_model['device_cat'].astype(str)

    # Formula: interaction between RunningView and DyslexiaIndicator
    formula = (
        'ReadingSpeed_wps ~ RunningView * DyslexiaIndicator '
        '+ C(device_cat) + gender + age + education + Flesch_Kincaid + img_width + num_words'
    )

    # Fit OLS without clustered cov initially; then attach cluster-robust covariance aligned to rows used in the model
    ols_res = smf.ols(formula, data=df_model).fit()

    # Align the participant_id values to the rows actually used by the fitted model
    # model.data.row_labels gives the original index labels of the rows used to build the design matrices
    try:
        row_labels = ols_res.model.data.row_labels
        # row_labels may be a list of index labels; use .loc to extract participant ids in the same order
        groups_aligned = pd.Categorical(df_model.loc[row_labels, 'participant_id']).codes
    except Exception:
        # Fallback: assume ols_res.model.endog has same order as df_model and use participant_id values directly
        groups_aligned = pd.Categorical(df_model['participant_id']).codes

    # Now get robust covariance results clustered by participant
    clustered_res = ols_res.get_robustcov_results(cov_type='cluster', groups=groups_aligned)

    # Return the clustered results object
    return clustered_res