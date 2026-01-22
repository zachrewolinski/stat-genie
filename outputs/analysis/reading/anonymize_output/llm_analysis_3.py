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
    # Make a copy to avoid modifying original
    df = df.copy()

    # Rename columns to meaningful names used in modeling
    # Assumes the input columns follow the provided schema
    # feature3: Reader View active (1/0)
    # feature5: time on page minus scrolling (milliseconds)
    # feature7: number of words on page
    # feature8: comprehension accuracy (proportion or fraction)
    # feature10: age
    # feature11: device
    # feature12: dyslexia severity (0/1/2)
    # feature16: retake (1/0)
    # feature17: dyslexia flag (1 = dyslexia, 0 = no)
    # feature18: native English ('Y'/'N')
    # feature19: readability score
    # feature14: gender numeric

    rename_map = {
        'feature3': 'ReaderView_raw',
        'feature5': 'ReadingTime_ms',
        'feature7': 'WordCount',
        'feature8': 'Comprehension_raw',
        'feature10': 'Age',
        'feature11': 'Device',
        'feature12': 'DyslexiaSeverity',
        'feature16': 'Retake_raw',
        'feature17': 'Dyslexia_raw',
        'feature18': 'LanguageNative_raw',
        'feature19': 'Readability',
        'feature14': 'Gender'
    }
    df = df.rename(columns=rename_map)

    # Basic type conversions
    df['ReaderView'] = pd.to_numeric(df.get('ReaderView_raw', df.get('feature3')), errors='coerce').astype('float')
    df['ReadingTime_ms'] = pd.to_numeric(df.get('ReadingTime_ms', df.get('feature5')), errors='coerce')
    df['WordCount'] = pd.to_numeric(df.get('WordCount', df.get('feature7')), errors='coerce')
    df['Comprehension'] = pd.to_numeric(df.get('Comprehension_raw', df.get('feature8')), errors='coerce')
    df['Age'] = pd.to_numeric(df.get('Age', df.get('feature10')), errors='coerce')
    df['Device'] = df.get('Device', df.get('feature11'))
    df['DyslexiaSeverity'] = pd.to_numeric(df.get('DyslexiaSeverity', df.get('feature12')), errors='coerce')
    df['Retake'] = pd.to_numeric(df.get('Retake_raw', df.get('feature16')), errors='coerce').astype('float')
    df['Dyslexia'] = pd.to_numeric(df.get('Dyslexia_raw', df.get('feature17')), errors='coerce').astype('float')
    df['LanguageNative'] = df.get('LanguageNative_raw', df.get('feature18'))
    df['Readability'] = pd.to_numeric(df.get('Readability', df.get('feature19')), errors='coerce')
    df['Gender'] = pd.to_numeric(df.get('Gender', df.get('feature14')), errors='coerce')

    # Convert language native flag to binary (1 = 'Y', 0 = otherwise)
    df['LanguageNative'] = df['LanguageNative'].map(lambda x: 1 if (isinstance(x, str) and x.upper() == 'Y') else (0 if pd.notnull(x) else np.nan))

    # Convert ReaderView and Dyslexia to binary {0,1}
    df['ReaderView'] = df['ReaderView'].map({1: 1, 0: 0}).astype('float')
    df['Dyslexia'] = df['Dyslexia'].map({1: 1, 0: 0}).astype('float')

    # Compute reading time in seconds and reading speed (words per second)
    df['ReadingTime_s'] = df['ReadingTime_ms'] / 1000.0
    df['ReadingSpeed_wps'] = df['WordCount'] / df['ReadingTime_s']

    # Remove rows with invalid or missing core measures
    df = df.dropna(subset=['ReadingTime_s', 'WordCount', 'ReadingSpeed_wps', 'ReaderView', 'Dyslexia'])

    # Remove implausible reading times or speeds (noise / non-reading interactions)
    # Keep speeds between 0.1 and 10 words per second (reasonable adult reading speeds roughly 1-4 wps; threshold widened)
    df = df[(df['ReadingSpeed_wps'] > 0.1) & (df['ReadingSpeed_wps'] < 10)]

    # If comprehension is missing, keep row but mark NaN (we'll drop later if desired)

    # Create device dummy variables (drop_first=True to use smartphone as baseline if present)
    device_dummies = pd.get_dummies(df['Device'].fillna('unknown').astype(str), prefix='Device', drop_first=True)
    df = pd.concat([df, device_dummies], axis=1)

    # Keep a minimal set of control columns; ensure columns exist even if absent in data
    # Ensure expected device dummies columns appear in the dataframe (if not present, they won't be used by the model code)

    # Interaction term between ReaderView and Dyslexia
    df['ReaderView_x_Dyslexia'] = df['ReaderView'] * df['Dyslexia']

    # Final selection: keep only columns needed for modeling and diagnostics
    keep_cols = [
        'ReaderView',
        'Dyslexia',
        'ReaderView_x_Dyslexia',
        'ReadingSpeed_wps',
        'ReadingTime_s',
        'WordCount',
        'Comprehension',
        'Age',
        'Readability',
        'Retake',
        'LanguageNative',
        'Gender'
    ]

    # Add any device dummy columns created
    keep_cols += [c for c in df.columns if c.startswith('Device_')]

    # Add DyslexiaSeverity for potential sensitivity checks
    if 'DyslexiaSeverity' in df.columns:
        keep_cols.append('DyslexiaSeverity')

    # Keep only these columns (and drop duplicates if any)
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    # Final drop of rows with missing dependent or primary iv/moderator
    df = df.dropna(subset=['ReadingSpeed_wps', 'ReaderView', 'Dyslexia'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Assumes df is already transformed by transform()
    # Build the design matrix for OLS with heteroskedasticity-robust SEs
    # Primary regressors: ReaderView, Dyslexia, and their interaction ReaderView_x_Dyslexia

    # Ensure required columns exist
    required = ['ReadingSpeed_wps', 'ReaderView', 'Dyslexia', 'ReaderView_x_Dyslexia']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns for modeling: ' + ','.join(missing))

    # Base controls
    control_cols = ['Age', 'Comprehension', 'Readability', 'Retake', 'LanguageNative', 'Gender']
    control_cols = [c for c in control_cols if c in df.columns]

    # Add device dummies if present (they start with 'Device_')
    device_cols = [c for c in df.columns if c.startswith('Device_')]

    # Compose regressors list
    regressors = ['ReaderView', 'Dyslexia', 'ReaderView_x_Dyslexia'] + control_cols + device_cols

    # Ensure regressors are numeric and fill missing controls with column mean (conservative)
    X = df[regressors].copy()
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors='coerce')
        # Impute missing control values with column mean (do not impute IVs/moderator)
        if X[col].isnull().any() and col not in ['ReaderView', 'Dyslexia', 'ReaderView_x_Dyslexia']:
            X[col] = X[col].fillna(X[col].mean())

    # Dependent variable
    y = df['ReadingSpeed_wps'].astype(float)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust standard errors (HC3)
    model = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted results object so caller can inspect params, summary, etc.
    return model


