from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/anonymize_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into an analysis-ready dataframe.

    Output columns (kept/created):
      - RecordID: original record identifier from feature1
      - PageID: page identifier from feature2 (categorical)
      - ReaderView: binary indicator (0/1) from feature3
      - ReadingTime_ms: reading time excluding scrolling (feature5)
      - ReadingTime_s: reading time in seconds
      - WordsCount: number of words on page (feature7)
      - ReadingSpeed_wps: words per second (WordsCount / ReadingTime_s)
      - ReadingSpeed_wps_wins: winsorized ReadingSpeed_wps at 1st and 99th pct
      - LogReadingSpeed: log(ReadingSpeed_wps_wins)
      - Dyslexia: binary dyslexia indicator (feature17 where available, fallback to feature12 mapping)
      - Age, Device, Retake, ComprehensionRate, NativeEnglish, FleschKincaid
    """
    df = df.copy()

    # Rename relevant columns for clarity
    rename_map = {
        'feature1': 'RecordID',
        'feature2': 'PageID',
        'feature3': 'ReaderView_raw',
        'feature4': 'TimeOnPage_ms',
        'feature5': 'ReadingTime_ms',
        'feature6': 'ScrollTime_ms',
        'feature7': 'WordsCount',
        'feature8': 'ComprehensionRate',
        'feature9': 'ImageWidth',
        'feature10': 'Age',
        'feature11': 'Device',
        'feature12': 'DyslexiaMulti',
        'feature13': 'Education',
        'feature14': 'Gender',
        'feature15': 'Language',
        'feature16': 'Retake',
        'feature17': 'Dyslexia',
        'feature18': 'NativeEnglish',
        'feature19': 'FleschKincaid',
        'feature20': 'feature20'
    }
    df.rename(columns=rename_map, inplace=True)

    # ReaderView: ensure binary 0/1
    df['ReaderView'] = pd.to_numeric(df.get('ReaderView_raw', 0), errors='coerce').fillna(0).astype(int)

    # Ensure ReadingTime_ms exists and is numeric; drop nonpositive times
    df['ReadingTime_ms'] = pd.to_numeric(df.get('ReadingTime_ms', np.nan), errors='coerce')
    df = df[df['ReadingTime_ms'].notnull()]
    df = df[df['ReadingTime_ms'] > 0]

    # WordsCount numeric
    df['WordsCount'] = pd.to_numeric(df.get('WordsCount', np.nan), errors='coerce')
    df = df[df['WordsCount'].notnull()]

    # Compute reading time in seconds and words per second
    df['ReadingTime_s'] = df['ReadingTime_ms'] / 1000.0
    df['ReadingSpeed_wps'] = df['WordsCount'] / df['ReadingTime_s']

    # Remove rows with non-finite speeds
    df = df[np.isfinite(df['ReadingSpeed_wps'])]

    # Winsorize ReadingSpeed at 1st and 99th percentiles to reduce influence of extreme outliers
    lower = df['ReadingSpeed_wps'].quantile(0.01)
    upper = df['ReadingSpeed_wps'].quantile(0.99)
    df['ReadingSpeed_wps_wins'] = df['ReadingSpeed_wps'].clip(lower=lower, upper=upper)

    # Log-transform the winsorized speed for modeling
    # Add a small constant guard (should not be needed because speed>0) but keep for numerical safety
    df['LogReadingSpeed'] = np.log(df['ReadingSpeed_wps_wins'] + 1e-9)

    # Dyslexia: prefer binary feature17; if missing, infer from feature12 (DyslexiaMulti: 0/1/2)
    if 'Dyslexia' in df.columns:
        df['Dyslexia'] = pd.to_numeric(df['Dyslexia'], errors='coerce')
    else:
        df['Dyslexia'] = np.nan
    if 'DyslexiaMulti' in df.columns:
        df['DyslexiaMulti'] = pd.to_numeric(df['DyslexiaMulti'], errors='coerce')
    # Where Dyslexia is missing, map DyslexiaMulti>0 to 1 (1 or 2 means some dyslexia)
    df['Dyslexia'] = df['Dyslexia'].fillna(df['DyslexiaMulti'].apply(lambda x: 1 if pd.notnull(x) and x in [1, 2] else np.nan))
    # Final binary cast (drop rows where dyslexia info missing)
    df = df[df['Dyslexia'].notnull()]
    df['Dyslexia'] = df['Dyslexia'].astype(int)

    # Other controls
    df['Age'] = pd.to_numeric(df.get('Age', np.nan), errors='coerce')
    df['Retake'] = pd.to_numeric(df.get('Retake', 0), errors='coerce').fillna(0).astype(int)
    df['ComprehensionRate'] = pd.to_numeric(df.get('ComprehensionRate', np.nan), errors='coerce')
    df['FleschKincaid'] = pd.to_numeric(df.get('FleschKincaid', df.get('FleschKincaid', np.nan)), errors='coerce')

    # NativeEnglish: convert 'Y'/'N' to 1/0 if necessary
    if 'NativeEnglish' in df.columns:
        df['NativeEnglish'] = df['NativeEnglish'].replace({'Y': 1, 'N': 0})
        # If still non-numeric, coerce
        df['NativeEnglish'] = pd.to_numeric(df['NativeEnglish'], errors='coerce')
    else:
        df['NativeEnglish'] = np.nan

    # Device and PageID ensure categorical
    if 'Device' in df.columns:
        df['Device'] = df['Device'].astype('category')
    if 'PageID' in df.columns:
        df['PageID'] = df['PageID'].astype('category')

    # Drop rows with missing values in key model columns
    required_cols = ['LogReadingSpeed', 'ReaderView', 'Dyslexia', 'ComprehensionRate', 'WordsCount']
    df = df.dropna(subset=required_cols)

    # Keep only the columns needed for modeling and a few diagnostics
    keep_cols = [
        'RecordID', 'PageID', 'ReaderView', 'ReadingTime_ms', 'ReadingTime_s', 'WordsCount',
        'ReadingSpeed_wps', 'ReadingSpeed_wps_wins', 'LogReadingSpeed', 'Dyslexia', 'Age', 'Device',
        'Retake', 'ComprehensionRate', 'NativeEnglish', 'FleschKincaid'
    ]
    # Some columns may not exist in the original; select those that do
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Reset index before returning
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit an OLS model predicting log reading speed with ReaderView, Dyslexia (moderator), their interaction,
    and control variables. Returns the fitted model object (statsmodels regression results).

    Model specification:
      LogReadingSpeed ~ ReaderView * Dyslexia + Age + Retake + ComprehensionRate + FleschKincaid + WordsCount
                       + C(Device) + C(PageID) + NativeEnglish

    Robust (HC3) standard errors are used to guard against heteroskedasticity.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure categorical variables are of type 'category' for formula interface
    if 'Device' in df.columns:
        df['Device'] = df['Device'].astype('category')
    if 'PageID' in df.columns:
        df['PageID'] = df['PageID'].astype('category')
    if 'NativeEnglish' in df.columns:
        # If native english is not binary, coerce to numeric
        df['NativeEnglish'] = pd.to_numeric(df['NativeEnglish'], errors='coerce')

    # Build formula. Use C(...) to treat Device and PageID as categorical fixed effects.
    formula = (
        'LogReadingSpeed ~ ReaderView * Dyslexia + Age + Retake + ComprehensionRate '
        '+ FleschKincaid + WordsCount + C(Device) + C(PageID) + NativeEnglish'
    )

    # Fit OLS with robust standard errors (HC3)
    model_res = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted results object. The caller can inspect model_res.summary() or model_res.params, etc.
    return model_res


