from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/replace_with_rvs_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables required for modeling the effect of Reader View on reading speed, and the interaction with dyslexia.

    Produces the following columns used in the model:
      - log_speed: natural log of speed (after removing zeros and extreme outliers)
      - reader_view: binary indicator (int)
      - dyslexia_bin: binary dyslexia indicator (0/1)
      - age, retake_trial, Flesch_Kincaid, num_words, img_width, device, page_id, uuid
      - english_native_Y: binary indicator (1 if english_native == 'Y')

    Notes:
      - Remove rows with missing essential fields.
      - Remove zero or negative speed values and trim extreme outliers by keeping between 1st and 99th percentile.
    """
    df = df.copy()

    # Ensure essential columns exist
    required_cols = [
        'uuid', 'speed', 'reader_view', 'dyslexia_bin', 'age', 'retake_trial',
        'Flesch_Kincaid', 'num_words', 'img_width', 'device', 'page_id', 'english_native'
    ]
    # If any required column missing, raise informative error
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing critical variables (presence check)
    df = df.dropna(subset=['uuid', 'speed', 'reader_view', 'dyslexia_bin', 'page_id'])

    # Ensure numeric types where appropriate
    # speed: coerce to numeric, then drop invalids
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed'])
    # Remove non-positive speeds
    df = df[df['speed'] > 0]

    # Other numeric controls
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce')
    df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    df['img_width'] = pd.to_numeric(df['img_width'], errors='coerce')

    # Robustly convert reader_view and dyslexia_bin to binary 0/1 integers.
    # Some datasets may encode these as strings or non-integer numbers.
    # We interpret any numeric value >= 1 as 1 (active/has dyslexia), else 0.
    # Coerce non-numeric to NaN, then treat NaN as 0 (conservative).
    rv_num = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0)
    df['reader_view'] = (rv_num >= 1).astype(int)

    dys_num = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0)
    df['dyslexia_bin'] = (dys_num >= 1).astype(int)

    # Trim extreme speed outliers to reduce influence of extreme values
    # Only compute quantiles if there are enough observations; otherwise skip trimming.
    if df['speed'].notna().sum() >= 2:
        q_low = df['speed'].quantile(0.01)
        q_high = df['speed'].quantile(0.99)
        # If quantiles are valid numbers, apply trimming
        if pd.notna(q_low) and pd.notna(q_high):
            df = df[(df['speed'] >= q_low) & (df['speed'] <= q_high)].copy()

    # Log-transform speed to stabilize variance
    df['log_speed'] = np.log(df['speed'])

    # Binary english native indicator
    df['english_native_Y'] = (df['english_native'] == 'Y').astype(int)

    # Device and page_id as categorical
    df['device'] = df['device'].astype('category')
    df['page_id'] = df['page_id'].astype('category')

    # Keep only columns necessary for modeling (but allow returning larger df). These exact columns are referenced in the model.
    model_cols = [
        'uuid', 'log_speed', 'speed', 'reader_view', 'dyslexia_bin', 'age', 'retake_trial',
        'Flesch_Kincaid', 'num_words', 'img_width', 'device', 'page_id', 'english_native_Y'
    ]

    # Some rows may have NaNs in control vars; drop those rows for the model
    df = df.dropna(subset=model_cols)

    # Ensure types are appropriate for modeling
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)
    df['retake_trial'] = df['retake_trial'].astype(int).where(df['retake_trial'].notna(), df['retake_trial'])

    # Return df containing at least the model columns (and any other original columns preserved)
    missing_after = [c for c in model_cols if c not in df.columns]
    if missing_after:
        raise KeyError(f"After transform, missing expected model columns: {missing_after}")

    # Reset index so downstream modeling receives a 0..n-1 index aligned with design matrices
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects model estimating the effect of Reader View on reading speed (log_speed),
    and its interaction with dyslexia (dyslexia_bin). A random intercept for participant (uuid)
    accounts for repeated measures.

    Model formula:
      log_speed ~ reader_view * dyslexia_bin + age + retake_trial + Flesch_Kincaid + num_words + img_width + C(device) + C(page_id) + english_native_Y

    Returns the fitted model object (statsmodels result) so the caller can inspect summary(), params, etc.
    """
    import statsmodels.formula.api as smf

    # Work on a copy and ensure a clean integer index so statsmodels' internal indexing aligns
    df = df.copy().reset_index(drop=True)

    # Check required columns exist
    required = [
        'log_speed', 'reader_view', 'dyslexia_bin', 'age', 'retake_trial',
        'Flesch_Kincaid', 'num_words', 'img_width', 'device', 'page_id', 'english_native_Y', 'uuid'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for modeling: {missing}")

    # Drop any rows with missing values in model inputs to avoid alignment issues
    df = df.dropna(subset=required).reset_index(drop=True)

    # Ensure categorical columns are category dtype (so C() behaves predictably)
    df['device'] = df['device'].astype('category')
    df['page_id'] = df['page_id'].astype('category')

    # Build formula. C(device) and C(page_id) treat those as categorical factors.
    formula = (
        'log_speed ~ reader_view * dyslexia_bin + age + retake_trial + '
        'Flesch_Kincaid + num_words + img_width + C(device) + C(page_id) + english_native_Y'
    )

    # Fit mixed-effects model with random intercept for each participant (uuid)
    # Use maximum likelihood (reml=False) for likelihood-based comparisons, but REML could also be used.
    md = smf.mixedlm(formula, df, groups=df['uuid'])
    mdf = md.fit(reml=False)

    # Return the fitted model results object
    return mdf