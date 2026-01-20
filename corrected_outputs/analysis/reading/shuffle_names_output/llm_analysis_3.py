from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/shuffle_names_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into an analysis-ready dataframe.

    Produces the following columns (used in the model):
      - ReaderView: binary 0/1 indicator whether Reader View was active
      - ReadingTime_ms: reading-only time in milliseconds (prefer 'language' column if available)
      - NumWords: number of words on page (prefer 'dyslexia' column which contains word counts)
      - ReadingSpeed_wps: words per second = NumWords / (ReadingTime_ms / 1000)
      - LogReadingSpeed: natural log of ReadingSpeed_wps (DV)
      - DyslexiaBin: dyslexia status (0/1/2)
      - Age, Gender, Education, FleschKincaid, ImgWidth, Device, PageID: control variables (copied/renamed)
    """
    df = df.copy()

    # Standardize key columns existence: assume columns from schema exist; guard missing gracefully
    # 1) Treatment indicator: prefer 'reader_view' ('Y'/'N'); fallback to 'running_time' (0/1)
    if 'reader_view' in df.columns:
        df['ReaderView'] = df['reader_view'].map({ 'Y': 1, 'N': 0 }).astype('float')
    else:
        df['ReaderView'] = np.nan

    if df['ReaderView'].isnull().all() and 'running_time' in df.columns:
        # running_time sometimes given as 1/0
        df['ReaderView'] = df['running_time'].astype(float)

    # 2) Reading time in milliseconds: prefer 'language' (schema indicates reading-only time) else try adjusted_running_time - scrolling_time
    df['ReadingTime_ms'] = np.nan
    if 'language' in df.columns:
        # use positive values only
        mask = df['language'].notna() & (df['language'] > 0)
        df.loc[mask, 'ReadingTime_ms'] = df.loc[mask, 'language']

    if df['ReadingTime_ms'].isna().any() and ('adjusted_running_time' in df.columns) and ('scrolling_time' in df.columns):
        # use adjusted_running_time minus scrolling_time as fallback
        fallback = df['adjusted_running_time'] - df['scrolling_time']
        mask = df['ReadingTime_ms'].isna() & fallback.notna() & (fallback > 0)
        df.loc[mask, 'ReadingTime_ms'] = fallback.loc[mask]

    # 3) NumWords: choose column that most likely contains the number of words ('dyslexia' appears to contain word counts per schema)
    if 'dyslexia' in df.columns:
        df['NumWords'] = pd.to_numeric(df['dyslexia'], errors='coerce')
    elif 'num_words' in df.columns:
        # fallback: some datasets encode words differently
        df['NumWords'] = pd.to_numeric(df['num_words'], errors='coerce')
    else:
        df['NumWords'] = np.nan

    # 4) Dyslexia status (moderator): prefer 'dyslexia_bin' column as described in schema
    if 'dyslexia_bin' in df.columns:
        df['DyslexiaBin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
    elif 'dyslexia' in df.columns and df['dyslexia'].nunique() <= 3:
        # if dyslexia column already small-coded
        df['DyslexiaBin'] = pd.to_numeric(df['dyslexia'], errors='coerce')
    else:
        df['DyslexiaBin'] = np.nan

    # 5) Controls: copy/rename to standardized columns if present
    for src, dst in [('age', 'Age'), ('gender', 'Gender'), ('education', 'Education'),
                     ('Flesch_Kincaid', 'FleschKincaid'), ('img_width', 'ImgWidth'),
                     ('device', 'Device'), ('page_id', 'PageID')]:
        if src in df.columns:
            df[dst] = df[src]
        else:
            df[dst] = np.nan

    # 6) Compute reading speed in words per second
    # Avoid division by zero or tiny reading times
    df['ReadingSpeed_wps'] = np.nan
    valid_mask = df['NumWords'].notna() & df['ReadingTime_ms'].notna() & (df['ReadingTime_ms'] > 100)  # require at least 100 ms
    df.loc[valid_mask, 'ReadingSpeed_wps'] = df.loc[valid_mask, 'NumWords'] / (df.loc[valid_mask, 'ReadingTime_ms'] / 1000.0)

    # 7) Remove or mark implausible speeds: zero/negative or extremely large (>30 words/sec considered implausible for natural reading)
    df.loc[(df['ReadingSpeed_wps'] <= 0) | (df['ReadingSpeed_wps'] > 30), 'ReadingSpeed_wps'] = np.nan

    # 8) Log-transform the DV
    df['LogReadingSpeed'] = np.nan
    df.loc[df['ReadingSpeed_wps'].notna(), 'LogReadingSpeed'] = np.log(df.loc[df['ReadingSpeed_wps'].notna(), 'ReadingSpeed_wps'])

    # 9) Final filtering: keep rows that have the DV, IV, and moderator
    keep_mask = df['LogReadingSpeed'].notna() & df['ReaderView'].notna() & df['DyslexiaBin'].notna()
    df = df.loc[keep_mask].copy()

    # 10) Ensure categorical codes are integer where relevant
    df['DyslexiaBin'] = df['DyslexiaBin'].astype(int)
    df['ReaderView'] = df['ReaderView'].astype(int)

    # 11) (Optional) drop any remaining rows with missing required controls (Age/Gender/Education) or leave NaNs to be handled by model
    # For modeling we will drop rows missing essential controls to avoid automatic listwise deletion surprises later
    required_controls = ['Age', 'Gender', 'Education', 'FleschKincaid', 'ImgWidth', 'Device', 'PageID']
    # If a majority of these are missing for a row, we still keep the row; here we require at least ReaderView, LogReadingSpeed, DyslexiaBin (already enforced)
    # but drop rows with missing Device or PageID because we use them for clustering / fixed effects
    df = df[df['PageID'].notna()]
    df = df[df['Device'].notna()]

    # Reset index and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating the effect of ReaderView on log reading speed, with DyslexiaBin as a moderator
    and standard controls. Returns the fitted statsmodels results object.

    Model specification:
      LogReadingSpeed ~ ReaderView * C(DyslexiaBin) + Age + Gender + Education + FleschKincaid + ImgWidth + C(Device) + C(PageID)

    We cluster standard errors by PageID to account for text-level dependence.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['LogReadingSpeed', 'ReaderView', 'DyslexiaBin', 'Age', 'Gender', 'Education',
                'FleschKincaid', 'ImgWidth', 'Device', 'PageID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula: interaction between ReaderView and categorical DyslexiaBin
    formula = 'LogReadingSpeed ~ ReaderView * C(DyslexiaBin) + Age + Gender + Education + FleschKincaid + ImgWidth + C(Device) + C(PageID)'

    # Fit OLS
    model = smf.ols(formula=formula, data=df)
    # Clustered standard errors by PageID (text-level clustering)
    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['PageID']})
    except Exception:
        # Fallback to conventional OLS if clustering fails
        results = model.fit()

    return results


