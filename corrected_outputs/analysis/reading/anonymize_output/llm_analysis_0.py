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
    """
    Transform the raw dataset into an analysis-ready dataframe.

    Produces these final columns used in the model:
      - ReaderView: binary 0/1 from feature3
      - ReadingTime_ms: numeric reading time excluding scrolling from feature5 (ms)
      - WordCount: number of words on the page from feature7
      - wpm: words per minute (derived)
      - log_wpm: natural log of wpm (dependent variable)
      - Dyslexia: binary 0/1 from feature17
      - Age, Device, Education, NativeEnglish, Comprehension, Flesch, Retake

    Filters unrealistic or missing observations.
    """
    df = df.copy()

    # Rename columns to meaningful names used downstream
    rename_map = {
        'feature3': 'ReaderView',         # 0/1
        'feature5': 'ReadingTime_ms',     # reading time excluding scrolling (ms)
        'feature7': 'WordCount',          # number of words on the page
        'feature8': 'Comprehension',      # comprehension accuracy (proportion)
        'feature10': 'Age',               # age in years
        'feature11': 'Device',            # device categorical
        'feature12': 'DyslexiaSeverity',  # optional severity (0/1/2)
        'feature13': 'Education',         # education categorical
        'feature16': 'Retake',            # 0/1
        'feature17': 'Dyslexia',          # binary 0/1 (has dyslexia)
        'feature18': 'NativeEnglish',     # 'Y'/'N'
        'feature19': 'Flesch'             # readability score
    }
    df = df.rename(columns=rename_map)

    # Required columns for initial processing
    required = ['ReaderView', 'ReadingTime_ms', 'WordCount', 'Dyslexia']

    # Coerce the required columns to numeric (may introduce NaN for non-numeric entries)
    for col in required:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing any required variable after coercion
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Now convert binary integer columns to plain numpy integer dtype (not pandas nullable Int64)
    if 'ReaderView' in df.columns:
        # safe because we dropped NA rows above
        df['ReaderView'] = df['ReaderView'].astype(np.int64)
    if 'Dyslexia' in df.columns:
        df['Dyslexia'] = df['Dyslexia'].astype(np.int64)

    # Ensure numeric types for the continuous required columns
    if 'ReadingTime_ms' in df.columns:
        df['ReadingTime_ms'] = pd.to_numeric(df['ReadingTime_ms'], errors='coerce')
    if 'WordCount' in df.columns:
        df['WordCount'] = pd.to_numeric(df['WordCount'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Remove implausible reading times or word counts (likely logging/measurement errors)
    if 'ReadingTime_ms' in df.columns:
        df = df[df['ReadingTime_ms'] > 200]   # exclude extremely short times (<200 ms)
    if 'WordCount' in df.columns:
        df = df[df['WordCount'] > 0]

    # Compute words per minute and log-transform the DV
    # Only compute if WordCount and ReadingTime_ms exist
    if ('WordCount' in df.columns) and ('ReadingTime_ms' in df.columns):
        df['wpm'] = df['WordCount'] * 60000.0 / df['ReadingTime_ms']
    else:
        df['wpm'] = np.nan

    # Remove extreme wpm values (implausible reading speed) to avoid undue influence
    if 'wpm' in df.columns:
        df = df[(df['wpm'] > 5) & (df['wpm'] < 1000)]

    # Log transform (natural log) to reduce skew and approximate normality for OLS
    df['log_wpm'] = np.log(df['wpm'])

    # Map NativeEnglish from 'Y'/'N' to 1/0; if not present leave as NA
    if 'NativeEnglish' in df.columns:
        df['NativeEnglish'] = df['NativeEnglish'].map({'Y': 1, 'N': 0})
        # ensure numeric type if mapping succeeded
        df['NativeEnglish'] = pd.to_numeric(df['NativeEnglish'], errors='coerce')

    # Ensure other controls are numeric / appropriate dtype if they exist
    for col in ['Comprehension', 'Age', 'Flesch', 'Retake']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert categorical controls to category dtype for use with C(...) in formulas
    if 'Device' in df.columns:
        df['Device'] = df['Device'].astype('category')
    if 'Education' in df.columns:
        df['Education'] = df['Education'].astype('category')

    # Drop rows missing the DV or primary IV/moderator after transformations
    df = df.dropna(subset=[c for c in ['log_wpm', 'ReaderView', 'Dyslexia'] if c in df.columns])

    # Keep only columns required for modeling (but preserve them as columns in final DF)
    final_cols = [
        'ReaderView', 'Dyslexia', 'log_wpm', 'wpm', 'ReadingTime_ms', 'WordCount',
        'Age', 'Device', 'Education', 'NativeEnglish', 'Comprehension', 'Flesch', 'Retake'
    ]
    # Keep whichever of these exist in df; this makes function robust if some optional controls are missing
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression testing whether ReaderView improves reading speed and whether that effect
    differs for readers with dyslexia (interaction ReaderView * Dyslexia).

    Model specification (primary):
      log_wpm ~ ReaderView * Dyslexia + Age + C(Device) + C(Education) + NativeEnglish + Comprehension + Flesch + WordCount + Retake

    Uses heteroskedasticity-robust standard errors (HC3).
    Returns the fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required = ['log_wpm', 'ReaderView', 'Dyslexia']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe passed to model().")

    # Build formula; include categorical controls only if they exist in the dataframe
    formula_terms = ['ReaderView * Dyslexia']
    if 'Age' in df.columns:
        formula_terms.append('Age')
    if 'Device' in df.columns:
        formula_terms.append('C(Device)')
    if 'Education' in df.columns:
        formula_terms.append('C(Education)')
    if 'NativeEnglish' in df.columns:
        formula_terms.append('NativeEnglish')
    if 'Comprehension' in df.columns:
        formula_terms.append('Comprehension')
    if 'Flesch' in df.columns:
        formula_terms.append('Flesch')
    if 'WordCount' in df.columns:
        formula_terms.append('WordCount')
    if 'Retake' in df.columns:
        formula_terms.append('Retake')

    formula = 'log_wpm ~ ' + ' + '.join(formula_terms)

    # Fit OLS with robust standard errors
    results = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Print a brief summary (user can inspect results object for full access)
    print(results.summary())

    return results