from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    Produces the following columns required for modeling:
      - ReaderView: binary (0/1) from feature3
      - Dyslexia: binary (0/1) from feature17 (1 indicates dyslexia)
      - reading_time_ms: numeric reading time excluding scrolling (feature5)
      - WordCount: number of words on page (feature7)
      - reading_speed_wps: words per second = WordCount / (reading_time_ms / 1000)
      - log_reading_speed: natural log of reading_speed_wps
      - Comprehension: feature8 (proportion correct)
      - Age: feature10
      - Device: feature11 (kept as categorical)
      - Language: feature15 (kept as categorical)
      - Readability: feature19
      - Retake: feature16 (0/1)
      - NativeEnglish: binary 1 if feature18 == 'Y', 0 if 'N'

    Rows with missing or invalid values for the above necessary fields are dropped.
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = [
        'feature3', 'feature5', 'feature7', 'feature8', 'feature10', 'feature11',
        'feature15', 'feature16', 'feature17', 'feature18', 'feature19'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Coerce numeric columns to numpy float dtype (not pandas nullable Float64)
    df['ReaderView'] = pd.to_numeric(df['feature3'], errors='coerce').astype(float)
    df['reading_time_ms'] = pd.to_numeric(df['feature5'], errors='coerce').astype(float)
    df['WordCount'] = pd.to_numeric(df['feature7'], errors='coerce').astype(float)
    df['Comprehension'] = pd.to_numeric(df['feature8'], errors='coerce').astype(float)
    df['Age'] = pd.to_numeric(df['feature10'], errors='coerce').astype(float)
    df['Readability'] = pd.to_numeric(df['feature19'], errors='coerce').astype(float)

    # Binary and categorical conversions
    df['Dyslexia'] = pd.to_numeric(df['feature17'], errors='coerce').astype(float)
    df['Retake'] = pd.to_numeric(df['feature16'], errors='coerce').astype(float)

    # Device and Language keep as original categorical values (strings)
    df['Device'] = df['feature11'].astype('category')
    df['Language'] = df['feature15'].astype('category')

    # Native English mapping: feature18 expected 'Y'/'N'
    df['NativeEnglish'] = df['feature18'].map({'Y': 1, 'N': 0}).astype(float)

    # Filter out rows with missing or invalid numeric values required for computing speed
    df = df.dropna(subset=['ReaderView', 'reading_time_ms', 'WordCount', 'Dyslexia', 'Comprehension'])

    # Remove rows with non-positive reading time or wordcount to avoid divide-by-zero
    df = df[df['reading_time_ms'] > 0]
    df = df[df['WordCount'] > 0]

    # Compute reading speed in words per second and log-transform
    df['reading_speed_wps'] = df['WordCount'] / (df['reading_time_ms'] / 1000.0)

    # Drop any rows with non-positive speed (shouldn't occur after filters, but safe)
    df = df[df['reading_speed_wps'] > 0]

    # Natural log transform to reduce skew
    df['log_reading_speed'] = np.log(df['reading_speed_wps']).astype(float)

    # Make sure binary columns are integer type (0/1)
    # At this point ReaderView and Dyslexia have no NaNs (dropped above), so safe to cast to int
    df['ReaderView'] = df['ReaderView'].astype(int)
    df['Dyslexia'] = df['Dyslexia'].astype(int)
    df['Retake'] = df['Retake'].fillna(0).astype(int)
    # NativeEnglish may have NaNs where language isn't Y/N; keep as numpy float64 with NaN preserved

    # Final selection: keep only columns used in modeling
    keep_cols = [
        'ReaderView', 'Dyslexia', 'reading_time_ms', 'WordCount', 'reading_speed_wps', 'log_reading_speed',
        'Comprehension', 'Age', 'Device', 'Language', 'Readability', 'Retake', 'NativeEnglish'
    ]
    df = df[keep_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression testing whether ReaderView improves reading speed and whether that effect
    differs for participants with dyslexia (interaction test).

    Model (on transformed dataframe produced by transform):
      log_reading_speed ~ ReaderView * Dyslexia + Comprehension + Age + Retake + NativeEnglish + Readability + WordCount + C(Device) + C(Language)

    Returns the fitted statsmodels regression results object (with robust standard errors).
    """
    # Ensure needed columns exist
    required = ['log_reading_speed', 'ReaderView', 'Dyslexia', 'Comprehension', 'Age', 'Retake', 'NativeEnglish', 'Readability', 'WordCount', 'Device', 'Language']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Build formula. Use categorical encoding for Device and Language via C(...).
    formula = (
        'log_reading_speed ~ ReaderView * Dyslexia + Comprehension + Age + Retake + NativeEnglish + Readability + WordCount '
        '+ C(Device) + C(Language)'
    )

    model = smf.ols(formula=formula, data=df)
    # Fit with heteroskedasticity-robust standard errors (HC3)
    # Using fit().get_robustcov_results is robust across statsmodels versions
    results = model.fit()
    results_robust = results.get_robustcov_results(cov_type='HC3')

    return results_robust