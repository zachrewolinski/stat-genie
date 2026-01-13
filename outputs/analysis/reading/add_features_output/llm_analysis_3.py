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
    Transform the raw dataset into the analysis dataframe. Produces the following final columns used in modeling:
      - LogWPM: log-transformed words-per-minute (dependent variable)
      - WPM: words-per-minute (kept for diagnostics)
      - ReaderView: binary indicator 0/1 from reader_view
      - Dyslexia: binary indicator 0/1 from dyslexia_bin (moderator)
      - Age, Device, EnglishNative, FleschKincaid, Retake, NumWords, PageID, UUID

    Steps:
      - Drop rows with missing or invalid adjusted_running_time or num_words
      - Compute WPM and LogWPM
      - Create / map control variables
      - Return a dataframe containing only the final columns
    """
    # Defensive copy
    df = df.copy()

    # Standardize column names we will use
    # Create analysis columns with exact names referenced in model
    df['ReaderView'] = df['reader_view'].astype(float)

    # Use the provided binary dyslexia indicator (dyslexia_bin) as the moderator
    # If dyslexia_bin isn't present but dyslexia is, we could fall back; here we use dyslexia_bin
    if 'dyslexia_bin' in df.columns:
        df['Dyslexia'] = df['dyslexia_bin'].astype(float)
    else:
        # fallback: treat dyslexia > 0 as dyslexia
        df['Dyslexia'] = df['dyslexia'].fillna(0).astype(float)

    # Map participant and page ids to final column names
    df['UUID'] = df['uuid']
    df['PageID'] = df['page_id']

    # Controls and other variables
    df['Age'] = df['age']
    df['Device'] = df['device']
    # Map english_native 'Y'/'N' to 1/0; if missing, set to 0 (non-native) to avoid dropping many rows
    if 'english_native' in df.columns:
        df['EnglishNative'] = df['english_native'].map({'Y': 1, 'N': 0})
        # If there are other encodings or NA, fill with 0
        df['EnglishNative'] = df['EnglishNative'].fillna(0).astype(float)
    else:
        df['EnglishNative'] = 0.0

    # Readability
    if 'Flesch_Kincaid' in df.columns:
        df['FleschKincaid'] = df['Flesch_Kincaid']
    elif 'Flesch_Kincaid ' in df.columns:
        # sometimes trailing spaces exist
        df['FleschKincaid'] = df['Flesch_Kincaid ']
    else:
        df['FleschKincaid'] = np.nan

    df['Retake'] = df['retake_trial'] if 'retake_trial' in df.columns else 0

    # Words and adjusted running time used to compute WPM
    df['NumWords'] = df['num_words']
    # Prefer adjusted_running_time (scrolling removed); fall back to running_time if not present
    if 'adjusted_running_time' in df.columns:
        df['AdjustedRunningTime'] = df['adjusted_running_time']
    else:
        df['AdjustedRunningTime'] = df['running_time']

    # Drop rows with missing or invalid values needed to compute WPM
    df = df.dropna(subset=['AdjustedRunningTime', 'NumWords', 'ReaderView', 'Dyslexia'])

    # Remove non-positive adjusted running time or num_words
    df = df[(df['AdjustedRunningTime'] > 0) & (df['NumWords'] > 0)]

    # Compute words-per-minute and log-transform
    # WPM = num_words * 60000 ms per minute / adjusted_running_time (ms)
    df['WPM'] = df['NumWords'] * 60000.0 / df['AdjustedRunningTime']

    # Drop rows where WPM is non-finite (defensive)
    df = df[np.isfinite(df['WPM'])]

    # Log transform to stabilize variance (natural log)
    # Add small constant to be defensive (shouldn't be necessary because we filtered >0)
    df['LogWPM'] = np.log(df['WPM'].clip(lower=1e-6))

    # Keep only the columns required for the model (in the exact names used by the model code)
    final_cols = [
        'LogWPM', 'WPM', 'ReaderView', 'Dyslexia', 'Age', 'Device', 'EnglishNative',
        'FleschKincaid', 'Retake', 'NumWords', 'PageID', 'UUID'
    ]

    # If some columns are missing (e.g., FleschKincaid), ensure they exist to avoid KeyErrors
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[final_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a linear model testing whether Reader View improves reading speed, and whether that effect differs
    for readers with dyslexia (interaction). We use log words-per-minute (LogWPM) as the dependent variable.

    Model specification (linear regression with cluster-robust SEs by participant UUID):
      LogWPM ~ ReaderView * Dyslexia + Age + C(Device) + EnglishNative + FleschKincaid + Retake + NumWords + C(PageID)

    We include C(Device) and C(PageID) as categorical fixed effects. Standard errors are clustered by UUID
    to account for within-participant correlations (multiple page observations per participant).

    Returns the fitted results object with cluster-robust covariance.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns present
    required = ['LogWPM', 'ReaderView', 'Dyslexia', 'Age', 'Device', 'EnglishNative',
                'FleschKincaid', 'Retake', 'NumWords', 'PageID', 'UUID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Drop rows with missing values in model variables
    model_df = df.dropna(subset=['LogWPM', 'ReaderView', 'Dyslexia', 'Age', 'Device', 'NumWords', 'UUID'])

    # Convert categorical columns to appropriate dtype for formula handling
    model_df['Device'] = model_df['Device'].astype('category')
    model_df['PageID'] = model_df['PageID'].astype('category')

    # Define formula with interaction between ReaderView and Dyslexia (moderation)
    formula = 'LogWPM ~ ReaderView * Dyslexia + Age + C(Device) + EnglishNative + FleschKincaid + Retake + NumWords + C(PageID)'

    # Fit OLS
    ols_res = smf.ols(formula=formula, data=model_df).fit()

    # Obtain cluster-robust standard errors clustered by participant UUID
    try:
        clustered_res = ols_res.get_robustcov_results(cov_type='cluster', groups=model_df['UUID'])
    except Exception:
        # If clustering fails for any reason, fall back to heteroskedasticity-robust (HC3)
        clustered_res = ols_res.get_robustcov_results(cov_type='HC3')

    # Print and return the clustered results object
    print(clustered_res.summary())
    return clustered_res


