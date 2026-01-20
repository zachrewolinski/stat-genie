from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/anonymize_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Original column mapping (for clarity):
    # feature3 -> ReaderView (0/1)
    # feature5 -> ReadingTime_ms (time on page minus scrolling) (ms)
    # feature4 -> TotalTime_ms (total time on page) (ms)
    # feature6 -> ScrollTime_ms (ms)
    # feature7 -> Words (number of words)
    # feature8 -> ComprehensionRate (proportion correct)
    # feature10 -> Age
    # feature11 -> Device
    # feature12 -> DyslexiaSeverity (0/1/2)
    # feature17 -> Dyslexia (0/1)
    # feature16 -> Retake (0/1)
    # feature18 -> NativeEnglish ('Y'/'N')
    # feature19 -> Flesch (readability)
    # feature2 -> PageID (page identifier)

    # Rename columns to meaningful names used in modeling
    rename_map = {
        'feature3': 'ReaderView',
        'feature5': 'ReadingTime_ms',
        'feature4': 'TotalTime_ms',
        'feature6': 'ScrollTime_ms',
        'feature7': 'Words',
        'feature8': 'ComprehensionRate',
        'feature10': 'Age',
        'feature11': 'Device',
        'feature12': 'DyslexiaSeverity',
        'feature17': 'Dyslexia',
        'feature16': 'Retake',
        'feature18': 'NativeEnglish',
        'feature19': 'Flesch',
        'feature2': 'PageID',
        'feature1': 'RecordID'
    }
    df = df.rename(columns=rename_map)

    # Cast types
    # ReaderView should be binary 0/1
    df['ReaderView'] = pd.to_numeric(df['ReaderView'], errors='coerce').astype('float')
    # Dyslexia present (0/1)
    df['Dyslexia'] = pd.to_numeric(df['Dyslexia'], errors='coerce').astype('float')
    # Reading time and words
    df['ReadingTime_ms'] = pd.to_numeric(df['ReadingTime_ms'], errors='coerce')
    df['Words'] = pd.to_numeric(df['Words'], errors='coerce')
    df['ComprehensionRate'] = pd.to_numeric(df['ComprehensionRate'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Flesch'] = pd.to_numeric(df['Flesch'], errors='coerce')
    df['Retake'] = pd.to_numeric(df['Retake'], errors='coerce').astype('float')

    # NativeEnglish -> binary (1 if 'Y', 0 otherwise)
    df['NativeEnglish'] = df['NativeEnglish'].astype(str).str.upper().map(lambda x: 1 if x == 'Y' else 0)

    # Device and PageID as categorical
    df['Device'] = df['Device'].astype('category')
    df['PageID'] = df['PageID'].astype('category')

    # Calculate reading speed in words per minute (WPM)
    # Use ReadingTime_ms (time on page minus scrolling) as the best available proxy for reading time.
    # Avoid division by zero or implausibly small reading times by filtering below.
    df['ReadingSpeed_wpm'] = np.nan
    valid_time_mask = (df['ReadingTime_ms'].notnull()) & (df['ReadingTime_ms'] > 300) & (df['Words'].notnull()) & (df['Words'] > 0)
    df.loc[valid_time_mask, 'ReadingSpeed_wpm'] = (df.loc[valid_time_mask, 'Words'] * 60000.0) / df.loc[valid_time_mask, 'ReadingTime_ms']

    # Remove infinite/zero/negative or extremely large speeds (likely artifacts)
    df.loc[~np.isfinite(df['ReadingSpeed_wpm']), 'ReadingSpeed_wpm'] = np.nan
    # Optionally, drop extreme outliers beyond the 99.9th percentile to stabilize estimation
    if df['ReadingSpeed_wpm'].notnull().sum() > 0:
        upper = df['ReadingSpeed_wpm'].quantile(0.999)
        lower = df['ReadingSpeed_wpm'].quantile(0.001)
        df.loc[(df['ReadingSpeed_wpm'] > upper) | (df['ReadingSpeed_wpm'] < lower), 'ReadingSpeed_wpm'] = np.nan

    # Log transform to reduce skew. Keep both raw WPM and log-WPM for reporting.
    df['LogReadingSpeed'] = np.nan
    mask_log = df['ReadingSpeed_wpm'].notnull() & (df['ReadingSpeed_wpm'] > 0)
    df.loc[mask_log, 'LogReadingSpeed'] = np.log(df.loc[mask_log, 'ReadingSpeed_wpm'])

    # Drop rows with missing key variables for the primary analysis
    df = df.dropna(subset=['ReaderView', 'Dyslexia', 'ReadingSpeed_wpm', 'LogReadingSpeed'])

    # Ensure binary columns are integer 0/1
    df['ReaderView'] = df['ReaderView'].astype(int)
    df['Dyslexia'] = df['Dyslexia'].astype(int)
    df['Retake'] = df['Retake'].fillna(0).astype(int)
    df['NativeEnglish'] = df['NativeEnglish'].fillna(0).astype(int)

    # For interpretability, create an interaction-ready categorical Moderator column (optional)
    # (We keep Dyslexia numeric 0/1 for modeling interaction.)

    # Final columns required for modeling are:
    # ['RecordID','ReaderView','ReadingTime_ms','TotalTime_ms','ScrollTime_ms','Words','ReadingSpeed_wpm','LogReadingSpeed',
    #  'Dyslexia','Age','Device','NativeEnglish','Flesch','Retake','ComprehensionRate','PageID']

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # We'll run an OLS regression on log(WPM) to estimate relative percent changes in reading speed.
    # The key term of interest is the interaction ReaderView:Dyslexia.

    import statsmodels.formula.api as smf

    # Make a local copy to avoid modifying the caller's DF
    data = df.copy()

    # Formula: LogReadingSpeed ~ ReaderView * Dyslexia + controls + categorical device + page fixed effects
    # Using C(Device) and C(PageID) includes dummy variables for device types and page fixed effects.
    formula = (
        'LogReadingSpeed ~ ReaderView * Dyslexia '
        '+ Age + Words + Flesch + NativeEnglish + Retake + ComprehensionRate '
        '+ C(Device) + C(PageID)'
    )

    # Fit OLS. Use cluster-robust standard errors clustered by PageID (page-level clustering) to account for page-level correlation.
    # If clustering by PageID is not appropriate in a dataset with singletons, fallback to HC3 robust errors.
    model = smf.ols(formula, data=data)

    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': data['PageID']})
    except Exception:
        # fallback to HC3 robust covariance if clustering fails
        results = model.fit(cov_type='HC3')

    # Return the fitted results object so the caller can inspect coefficients, p-values, and diagnostics.
    return results


