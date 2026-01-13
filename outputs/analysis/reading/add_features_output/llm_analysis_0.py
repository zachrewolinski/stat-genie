from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/add_features_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into the analysis dataframe.
    Produces these columns (used by the model):
      - uuid, page_id, reader_view, log_speed, is_dyslexic,
        num_words, age, device, Flesch_Kincaid, english_native, retake_trial

    Steps:
      - drop rows missing the core variables
      - construct an explicit binary dyslexia indicator (is_dyslexic)
      - compute log_speed = log(speed) to reduce skew
      - keep only columns required for modeling
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['speed', 'reader_view', 'num_words', 'page_id', 'uuid']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in dataframe")

    # Prefer an existing dyslexia binary if present, otherwise derive from 'dyslexia'
    if 'dyslexia_bin' in df.columns:
        df['is_dyslexic'] = df['dyslexia_bin'].astype(float).fillna(0).astype(int)
    elif 'dyslexia' in df.columns:
        # dyslexia: 0=no, 1=dyslexia, 2=severe -> treat any >=1 as dyslexic
        df['is_dyslexic'] = (df['dyslexia'].fillna(0) >= 1).astype(int)
    else:
        # if no dyslexia indicator available, create a column of zeros (no dyslexia info)
        df['is_dyslexic'] = 0

    # Drop rows with missing values in the key variables used in the model
    keep_cols = ['speed', 'reader_view', 'is_dyslexic', 'num_words', 'age', 'device',
                 'Flesch_Kincaid', 'english_native', 'retake_trial', 'page_id', 'uuid']
    # Only drop NAs in columns that exist in the dataset; otherwise keep the row (we already created defaults where necessary)
    cols_to_check = [c for c in keep_cols if c in df.columns]
    df = df.dropna(subset=cols_to_check)

    # Filter unreasonable speed values (non-positive) and compute log speed
    # Add small epsilon to avoid log(0) if there are extremely small positive values
    df = df[df['speed'] > 0]
    df['log_speed'] = np.log(df['speed'].astype(float).clip(lower=1e-6))

    # Ensure types: reader_view should be numeric (0/1)
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').astype(int)

    # Keep only the columns needed for modeling (this is our final dataframe)
    final_cols = [c for c in ['uuid', 'page_id', 'reader_view', 'log_speed', 'is_dyslexic', 'num_words', 'age', 'device', 'Flesch_Kincaid', 'english_native', 'retake_trial'] if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model of log reading speed on Reader View, moderation by dyslexia,
    and controls. We include categorical fixed effects for device and page_id and
    cluster standard errors by participant uuid to account for within-subject correlation.

    Model formula:
      log_speed ~ reader_view * is_dyslexic + C(device) + num_words + age + Flesch_Kincaid + C(english_native) + retake_trial + C(page_id)

    Returns the fitted results object (statsmodels regression results).
    """
    import statsmodels.formula.api as smf

    # Verify required columns
    model_cols = ['log_speed', 'reader_view', 'is_dyslexic', 'num_words', 'age', 'device', 'Flesch_Kincaid', 'english_native', 'retake_trial', 'page_id', 'uuid']
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe is missing required columns for model: {missing}")

    # Build formula. Use categorical encoding for device, english_native and page_id
    formula = ('log_speed ~ reader_view * is_dyslexic + C(device) + num_words + age '
               '+ Flesch_Kincaid + C(english_native) + retake_trial + C(page_id)')

    # Fit OLS and compute cluster-robust standard errors by participant (uuid)
    model = smf.ols(formula, data=df)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['uuid']})

    # Return the fitted results object so the caller can inspect coefficients, summary, confidence intervals, etc.
    return results


