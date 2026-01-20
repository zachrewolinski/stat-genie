from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Optional top-level read (kept from original file; can be ignored by callers)
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/anonymize_output/reading.csv')
except Exception:
    # If the environment does not have the file, ignore the error so the module can still be imported.
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    Produces these columns used in the model (exact names must match the contract):
      - ReaderView (0/1)
      - Dyslexia (0/1)
      - ReadingTime_ms (feature5)
      - Words (feature7)
      - ReadingSpeed_wps (Words per second)
      - Comprehension (feature8)
      - Age (feature10)
      - Device (feature11)
      - NativeEnglish (1 if feature18 == 'Y', else 0)
      - Retake (feature16)
      - Flesch (feature19)

    Filters out invalid/zero reading times or zero words and extreme outliers in ReadingSpeed_wps.
    """
    df = df.copy()

    # Numeric conversions for source features (coerce errors to NaN)
    df['ReaderView'] = pd.to_numeric(df.get('feature3'), errors='coerce')  # will be converted to int after NA removal
    df['ReadingTime_ms'] = pd.to_numeric(df.get('feature5'), errors='coerce')
    df['Words'] = pd.to_numeric(df.get('feature7'), errors='coerce')
    df['Comprehension'] = pd.to_numeric(df.get('feature8'), errors='coerce')
    df['Age'] = pd.to_numeric(df.get('feature10'), errors='coerce')
    # Device as categorical (keep original labels)
    if 'feature11' in df.columns:
        df['Device'] = df['feature11'].astype('category')
    else:
        df['Device'] = pd.Categorical([None] * len(df))
    df['Retake'] = pd.to_numeric(df.get('feature16'), errors='coerce')
    df['Flesch'] = pd.to_numeric(df.get('feature19'), errors='coerce')

    # Native English: map 'Y'/'N' to 1/0
    if 'feature18' in df.columns:
        native_map = df['feature18'].astype(str).str.upper().map({'Y': 1, 'N': 0})
        df['NativeEnglish'] = pd.to_numeric(native_map, errors='coerce')
    else:
        df['NativeEnglish'] = np.nan

    # Dyslexia: prefer feature12 (0=no,1=dyslexia,2=severe). If missing, fall back to feature17 (0/1)
    df['Dyslexia'] = np.nan
    if 'feature12' in df.columns:
        feat12 = pd.to_numeric(df['feature12'], errors='coerce')
        mask12 = feat12.notna()
        # binary: any value >=1 indicates dyslexia/severe
        df.loc[mask12, 'Dyslexia'] = (feat12[mask12] >= 1).astype(float)
    if 'feature17' in df.columns:
        # Only use feature17 where Dyslexia is still missing
        missing_mask = df['Dyslexia'].isna()
        feat17 = pd.to_numeric(df.loc[missing_mask, 'feature17'], errors='coerce')
        mask17 = feat17.notna()
        df.loc[missing_mask & mask17, 'Dyslexia'] = (feat17[mask17] == 1).astype(float)

    # Compute reading speed (words per second). Exclude non-positive times/words.
    df['ReadingSpeed_wps'] = np.nan
    valid_time_words = (
        df['ReadingTime_ms'].notna()
        & df['Words'].notna()
        & (df['ReadingTime_ms'] > 0)
        & (df['Words'] > 0)
    )
    df.loc[valid_time_words, 'ReadingSpeed_wps'] = (
        df.loc[valid_time_words, 'Words'] * 1000.0 / df.loc[valid_time_words, 'ReadingTime_ms']
    )

    # Drop rows with missing values in the core analysis columns
    keep_cols = [
        'ReaderView',
        'Dyslexia',
        'ReadingTime_ms',
        'Words',
        'ReadingSpeed_wps',
        'Comprehension',
        'Age',
        'Device',
        'NativeEnglish',
        'Retake',
        'Flesch',
    ]
    # Ensure all keep_cols exist in df before calling dropna
    missing_keep = [c for c in keep_cols if c not in df.columns]
    if missing_keep:
        raise ValueError(f"Missing required source columns for transform: {missing_keep}")

    df = df.dropna(subset=keep_cols)

    # Convert dtypes to numpy-backed dtypes that statsmodels/patsy can handle
    # Integer indicators: ReaderView, Dyslexia, NativeEnglish, Retake
    for col in ['ReaderView', 'Dyslexia', 'NativeEnglish', 'Retake']:
        # Values should be 0/1; cast to int64
        df[col] = df[col].astype(int)

    # Continuous variables to float64
    for col in ['ReadingTime_ms', 'Words', 'ReadingSpeed_wps', 'Comprehension', 'Age', 'Flesch']:
        df[col] = df[col].astype(float)

    # Ensure Device is a categorical or object type acceptable to patsy
    if not pd.api.types.is_categorical_dtype(df['Device']):
        df['Device'] = df['Device'].astype('category')

    # Remove extreme outliers in ReadingSpeed_wps to reduce influence of measurement errors
    # Keep observations between the 0.5th and 99.5th percentiles
    lower = df['ReadingSpeed_wps'].quantile(0.005)
    upper = df['ReadingSpeed_wps'].quantile(0.995)
    df = df[(df['ReadingSpeed_wps'] >= lower) & (df['ReadingSpeed_wps'] <= upper)]

    # Reset index and return only the columns needed for the model (plus ReadingTime_ms for diagnostics)
    result_cols = [
        'ReaderView',
        'Dyslexia',
        'ReadingTime_ms',
        'Words',
        'ReadingSpeed_wps',
        'Comprehension',
        'Age',
        'Device',
        'NativeEnglish',
        'Retake',
        'Flesch',
    ]
    df = df.loc[:, result_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model testing whether ReaderView improves reading speed, and whether that effect is moderated by dyslexia.

    Model specification:
      ReadingSpeed_wps ~ ReaderView * Dyslexia + Age + C(Device) + NativeEnglish + Retake + Flesch + Comprehension + Words

    Returns the fitted statsmodels regression results object with robust (HC3) standard errors.
    """
    # Ensure required columns are present
    required = [
        'ReadingSpeed_wps',
        'ReaderView',
        'Dyslexia',
        'Age',
        'Device',
        'NativeEnglish',
        'Retake',
        'Flesch',
        'Comprehension',
        'Words',
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop any remaining NAs in model variables
    model_df = df.dropna(subset=required).copy()

    # Ensure dtypes are appropriate for patsy/statsmodels (numpy dtypes or pandas categorical)
    # Convert indicator columns to plain int64, continuous to float64
    for col in ['ReaderView', 'Dyslexia', 'NativeEnglish', 'Retake']:
        model_df[col] = model_df[col].astype(int)
    for col in ['ReadingSpeed_wps', 'Age', 'Flesch', 'Comprehension', 'Words', 'ReadingTime_ms']:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype(float)
    if not pd.api.types.is_categorical_dtype(model_df['Device']):
        model_df['Device'] = model_df['Device'].astype('category')

    # Build formula. Device is treated as categorical using C(Device).
    formula = 'ReadingSpeed_wps ~ ReaderView * Dyslexia + Age + C(Device) + NativeEnglish + Retake + Flesch + Comprehension + Words'

    # Fit OLS
    fit = smf.ols(formula=formula, data=model_df).fit()

    # Also compute robust (HC3) covariance
    fit_robust = fit.get_robustcov_results(cov_type='HC3')

    # Return the robust-fit object (has .summary(), .params, etc.)
    return fit_robust