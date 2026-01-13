from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Optional top-level CSV read; transform should operate on any passed DataFrame.
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/add_features_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for modeling.
    Produces the following columns required by the model:
      - reader_view (0/1)
      - dyslexia_bin (0/1)
      - log_speed (log of capped speed)
      - num_words, Flesch_Kincaid, age, retake_trial, english_native_bin, correct_rate
      - device (categorical), page_id (categorical), uuid (participant id)

    Approach:
      - Drop rows missing the outcome or core predictors
      - Cap speed at the 99th percentile to reduce extreme outliers, then log-transform
      - Map english_native to binary
      - Impute remaining numeric control missings with column median
    """
    df = df.copy()

    # Ensure key columns exist
    required = ['speed', 'reader_view', 'dyslexia_bin', 'num_words', 'page_id', 'uuid']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not in dataframe")

    # Drop rows missing DV or the primary IV or moderator
    df = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin', 'num_words', 'page_id', 'uuid'])

    # Convert types / basic recoding
    # reader_view may be boolean/0/1; coerce to int safely
    # Use to_numeric but handle booleans/strings robustly by first mapping truthy values
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)

    # dyslexia_bin should be 0/1 already; coerce to int (if it's float)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0).astype(int)

    if 'retake_trial' in df.columns:
        # fill missing retake flags with 0 and cast to int
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Map english_native to binary column english_native_bin (Y -> 1, else 0)
    if 'english_native' in df.columns:
        # handle various representations robustly
        eng_series = df['english_native'].fillna('').astype(str).str.upper().str.strip()
        df['english_native_bin'] = (eng_series == 'Y').astype(int)
    else:
        df['english_native_bin'] = 0

    # Cap extremely large speeds at the 99th percentile to reduce influence of outliers, then log-transform.
    # We add 1 before log to avoid issues with zero.
    upper = df['speed'].quantile(0.99)
    df['speed_capped'] = df['speed'].clip(upper=upper)
    # Guard against negative speeds: replace negatives with small positive number before log
    df['speed_capped'] = df['speed_capped'].where(df['speed_capped'] >= 0, 0.0)
    df['log_speed'] = np.log(df['speed_capped'] + 1)

    # Ensure numeric control columns exist; if missing, create with median or sensible default
    numeric_controls = ['Flesch_Kincaid', 'age', 'correct_rate']
    for col in numeric_controls:
        if col not in df.columns:
            df[col] = np.nan

    # Impute remaining numeric missings with column median (so model can run)
    impute_cols = ['num_words', 'Flesch_Kincaid', 'age', 'correct_rate']
    for c in impute_cols:
        if c in df.columns:
            median_val = pd.to_numeric(df[c], errors='coerce').median()
            # If median is nan (all values missing), replace with 0 to avoid leaving NaNs
            if pd.isna(median_val):
                median_val = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(median_val)

    # Device and page id -> ensure categorical
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    else:
        df['device'] = 'unknown'
        df['device'] = df['device'].astype('category')

    df['page_id'] = df['page_id'].astype('category')

    # Keep only columns needed for modeling (but keep originals uuid and speed for reference)
    needed = [
        'uuid', 'page_id', 'device', 'reader_view', 'dyslexia_bin',
        'speed', 'speed_capped', 'log_speed', 'num_words', 'Flesch_Kincaid',
        'age', 'retake_trial', 'english_native_bin', 'correct_rate'
    ]
    # Guarantee all needed columns present
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    df = df[needed]

    # Final drop of any rows with NA in model-critical columns
    model_critical = ['log_speed', 'reader_view', 'dyslexia_bin', 'num_words', 'page_id', 'uuid']
    df = df.dropna(subset=model_critical)

    # Ensure uuid is treated consistently (keep as-is for record, but no further action needed here)
    # Ensure uuid index alignment is preserved (keep as column)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fits an OLS model testing whether Reader View improves reading speed and whether that effect
    differs for readers with dyslexia (interaction). Uses clustered robust SEs at the participant (uuid) level.

    Model formula:
      log_speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + age + retake_trial
                   + english_native_bin + correct_rate + C(device) + C(page_id)

    Returns the fitted results object (statsmodels RegressionResultsWrapper).
    """
    # Ensure required columns present
    cols_needed = ['log_speed', 'reader_view', 'dyslexia_bin', 'num_words', 'Flesch_Kincaid',
                   'age', 'retake_trial', 'english_native_bin', 'correct_rate', 'device', 'page_id', 'uuid']
    for c in cols_needed:
        if c not in df.columns:
            raise KeyError(f"Required column for modeling '{c}' not present in dataframe")

    # Work on a copy and reset the index to ensure alignment between data and cluster groups
    data = df.copy().reset_index(drop=True)

    # Define formula with interaction between reader_view and dyslexia_bin
    formula = (
        'log_speed ~ reader_view * dyslexia_bin + '
        'num_words + Flesch_Kincaid + age + retake_trial + english_native_bin + correct_rate + '
        'C(device) + C(page_id)'
    )

    # Fit OLS without cluster covariance first to let statsmodels determine which rows are used.
    model_fit = smf.ols(formula=formula, data=data)
    results_ols = model_fit.fit()

    # Align cluster groups to the rows actually used in the fitted model.
    # results_ols.model.data.row_labels contains the index labels (from `data`) of the observations used.
    row_labels = results_ols.model.data.row_labels
    # Create a Series of uuid indexed by data.index, then reindex to the row_labels to match rows used.
    groups_aligned = pd.Series(data['uuid'].values, index=data.index).reindex(row_labels)

    if groups_aligned.isnull().any():
        # If any group is missing after reindexing, that indicates a mismatch; raise a clear error.
        raise ValueError("Mismatch when aligning cluster groups to model rows; some groups are missing after reindexing.")

    # Convert group labels to consecutive integer codes required by statsmodels' cluster routines.
    group_codes = pd.Categorical(groups_aligned).codes

    # Obtain cluster-robust covariance results using the aligned integer group codes.
    results_cluster = results_ols.get_robustcov_results(cov_type='cluster', groups=group_codes)

    return results_cluster