from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Optional top-level read (kept for compatibility; can be removed if not needed)
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/noperturb_output/reading.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataframe into the analysis-ready dataframe.

    Ensures the FINAL dataframe contains the exact columns required by the
    analysis (see conceptual variables). Fills or imputes missing controls
    conservatively so that the modeling step does not fail due to mismatched
    row dropping between patsy and an externally supplied groups vector.

    Required final columns (ensured by this function):
      - reader_view (0/1 int)
      - log_speed (float)
      - dyslexia_bin (0/1 int)
      - num_words_c, Flesch_Kincaid_c, age_c, img_width_c (floats, mean-centered or 0)
      - device (categorical)
      - page_id (categorical)
      - uuid (identifier; if missing, populated with unique per-row ids)
      - english_native (numeric 0/1)
      - retake_trial (0/1 int)
      - correct_rate (float, filled with mean if missing)
    """
    df = df.copy()

    # Create or coerce dyslexia_bin: prefer existing column, otherwise derive from 'dyslexia'
    if 'dyslexia_bin' in df.columns:
        df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
    else:
        if 'dyslexia' in df.columns:
            df['dyslexia_bin'] = np.where(pd.to_numeric(df['dyslexia'], errors='coerce') >= 1, 1, 0)
        else:
            df['dyslexia_bin'] = np.nan

    # Ensure reader_view exists and is integer 0/1
    if 'reader_view' in df.columns:
        df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)
    else:
        df['reader_view'] = 0

    # Speed: numeric, drop rows with missing speed/reader_view/dyslexia_bin later
    df['speed'] = pd.to_numeric(df.get('speed', pd.Series(dtype=float)), errors='coerce')

    # Defensive: correct_rate
    if 'correct_rate' in df.columns:
        df['correct_rate'] = pd.to_numeric(df['correct_rate'], errors='coerce')
    else:
        df['correct_rate'] = np.nan

    # retake_trial
    if 'retake_trial' in df.columns:
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Clip speed to avoid log(0) and extreme negatives; then log-transform
    df['speed'] = df['speed'].clip(lower=1e-3)
    df['log_speed'] = np.log(df['speed'])

    # Mean-center continuous predictors if present, otherwise create zero column
    continuous_to_center = ['num_words', 'Flesch_Kincaid', 'age', 'img_width']
    for col in continuous_to_center:
        centered = col + '_c'
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            mean_val = numeric_col.mean()
            if np.isnan(mean_val):
                # If mean is NaN (all values missing), create zero column
                df[centered] = 0.0
            else:
                df[centered] = numeric_col.fillna(mean_val) - mean_val
        else:
            # create zero column to avoid NaNs so patsy won't drop rows
            df[centered] = 0.0

    # Device, page_id, uuid: ensure categorical identifiers exist and have no missing values
    # Fill missing with a stable placeholder ('unknown') except uuid where we create unique ids for missing values
    if 'device' in df.columns:
        df['device'] = df['device'].fillna('unknown').astype(str).astype('category')
    else:
        df['device'] = pd.Series(['unknown'] * len(df), dtype='category')

    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].fillna('unknown').astype(str).astype('category')
    else:
        df['page_id'] = pd.Series(['unknown'] * len(df), dtype='category')

    if 'uuid' in df.columns:
        # If there are missing uuids, assign unique placeholder ids per-row to avoid NA groups
        missing_mask = pd.isna(df['uuid'])
        if missing_mask.any():
            # create unique ids for missing entries to preserve row alignment
            replacement_ids = ['missing_uuid_{}'.format(i) for i in df.index[missing_mask]]
            df.loc[missing_mask, 'uuid'] = replacement_ids
        # ensure string dtype (groups may be strings)
        df['uuid'] = df['uuid'].astype(str)
    else:
        # create unique uuid per row to ensure grouping column exists
        df['uuid'] = ['generated_uuid_{}'.format(i) for i in df.index]
        df['uuid'] = df['uuid'].astype(str)

    # English_native: prefer numeric 0/1; if missing, fill with 0 (conservative)
    if 'english_native' in df.columns:
        # try to coerce to numeric; if non-numeric categories (yes/no) try mapping common patterns
        en = df['english_native']
        # First attempt numeric coercion
        en_num = pd.to_numeric(en, errors='coerce')
        if en_num.isnull().all():
            # Try mapping common textual values
            en_str = en.astype(str).str.lower().str.strip()
            mapped = en_str.replace({'yes': 1, 'y': 1, 'true': 1, 't': 1,
                                     'no': 0, 'n': 0, 'false': 0, 'f': 0})
            mapped_num = pd.to_numeric(mapped, errors='coerce')
            if mapped_num.isnull().all():
                df['english_native'] = 0
            else:
                df['english_native'] = mapped_num.fillna(0).astype(int)
        else:
            df['english_native'] = en_num.fillna(0).astype(int)
    else:
        df['english_native'] = 0

    # Ensure dyslexia_bin and reader_view are numeric 0/1
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0).astype(int)
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)

    # If correct_rate has missing values, fill with column mean (conservative)
    if df['correct_rate'].isnull().any():
        mean_corr = df['correct_rate'].mean()
        if np.isnan(mean_corr):
            mean_corr = 0.0
        df['correct_rate'] = df['correct_rate'].fillna(mean_corr)

    # Ensure retake_trial is integer 0/1
    df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)

    # Final drop: ensure rows used by the model have no missing values for core columns
    required_cols = [
        'log_speed', 'reader_view', 'dyslexia_bin', 'num_words_c',
        'Flesch_Kincaid_c', 'age_c', 'device', 'english_native',
        'retake_trial', 'correct_rate', 'img_width_c', 'page_id', 'uuid'
    ]

    # In the unlikely event any required column still has missing values, drop those rows.
    df = df.dropna(subset=required_cols)

    # Reset index to ensure contiguous integer indexing (required by statsmodels MixedLM)
    df = df.reset_index(drop=True)

    # Return transformed dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects linear model to estimate the effect of Reader View on reading speed,
    and test whether that effect differs by dyslexia status (interaction).

    Model (formula):
      log_speed ~ reader_view * dyslexia_bin + num_words_c + Flesch_Kincaid_c + age_c
                   + C(device) + english_native + retake_trial + correct_rate + img_width_c + C(page_id)

    Random effects: random intercept for participant (uuid) to account for repeated measures.

    Returns the fitted model results object (statsmodels MixedLMResults).
    """
    # Formula includes interaction to test whether Reader View effect differs for dyslexic readers
    formula = (
        'log_speed ~ reader_view * dyslexia_bin + num_words_c + Flesch_Kincaid_c + age_c '
        '+ C(device) + english_native + retake_trial + correct_rate + img_width_c + C(page_id)'
    )

    # Ensure group column exists
    if 'uuid' not in df.columns:
        raise ValueError("'uuid' column is required for the mixed effects grouping but is missing from the dataframe.")

    # Ensure contiguous integer index for statsmodels
    df = df.reset_index(drop=True)

    # Fit mixed effects model with random intercept for each participant
    # At this point transform() should have ensured there are no missing values
    md = smf.mixedlm(formula, df, groups=df['uuid'], re_formula='~1')
    mdf = md.fit(reml=False)

    # Return the fitted model object (contains params, pvalues, summary(), etc.)
    return mdf