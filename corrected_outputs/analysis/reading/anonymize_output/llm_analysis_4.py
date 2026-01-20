from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/anonymize_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset to analytic dataframe with the exact columns used in the statistical model.

    Final columns created (and used later in the model):
      - ReaderView (0/1)
      - ReadingTime_ms, ReadingTime_s (intermediate)
      - NumWords
      - ReadingSpeed_wps (DV)
      - Dyslexia (0/1)
      - DyslexiaSeverity (0/1/2)
      - Age, Device, Comprehension, IsNativeEnglish, Education, Gender, FKScore, IsRetake

    The function performs type conversion, basic sanity filtering (remove zero/very small reading times and extreme outliers), and drops rows with missing values in required variables.
    """
    df = df.copy()

    # Create clear column names used in modeling
    df['ReaderView'] = pd.to_numeric(df.get('feature3'), errors='coerce')
    df['ReadingTime_ms'] = pd.to_numeric(df.get('feature5'), errors='coerce')
    df['NumWords'] = pd.to_numeric(df.get('feature7'), errors='coerce')
    df['Comprehension'] = pd.to_numeric(df.get('feature8'), errors='coerce')

    # compute reading time in seconds and reading speed (words per second)
    df['ReadingTime_s'] = df['ReadingTime_ms'] / 1000.0
    # Avoid division by zero / invalid values
    df['ReadingSpeed_wps'] = np.where(df['ReadingTime_s'] > 0, df['NumWords'] / df['ReadingTime_s'], np.nan)

    # Dyslexia indicators
    # feature17 is a binary indicator (0/1) for dyslexia according to the schema
    df['Dyslexia'] = pd.to_numeric(df.get('feature17'), errors='coerce').fillna(0)
    # feature12 contains 0/1/2 dyslexia severity
    df['DyslexiaSeverity'] = pd.to_numeric(df.get('feature12'), errors='coerce')

    # Other covariates
    df['Age'] = pd.to_numeric(df.get('feature10'), errors='coerce')
    # keep Device as string (will be converted to categorical in model)
    df['Device'] = df.get('feature11').astype(str)
    # Map native English: feature18 'Y'/'N'
    df['IsNativeEnglish'] = df.get('feature18').map({'Y': 1, 'N': 0})
    df['IsNativeEnglish'] = pd.to_numeric(df['IsNativeEnglish'], errors='coerce')
    df['Education'] = df.get('feature13').astype(str)
    df['Gender'] = pd.to_numeric(df.get('feature14'), errors='coerce')
    df['FKScore'] = pd.to_numeric(df.get('feature19'), errors='coerce')
    df['IsRetake'] = pd.to_numeric(df.get('feature16'), errors='coerce')

    # Basic sanity filtering for reading time: remove impossible/implausible values
    # Remove non-positive or extremely small reading times (<= 0.05 s) and extremely large times (> 5 min = 300s)
    df.loc[df['ReadingTime_s'] <= 0.05, 'ReadingSpeed_wps'] = np.nan
    df.loc[df['ReadingTime_s'] > 300, 'ReadingSpeed_wps'] = np.nan

    # Remove entries with missing critical variables
    required_cols = [
        'ReadingSpeed_wps', 'ReaderView', 'Dyslexia', 'NumWords', 'ReadingTime_ms'
    ]
    df = df.dropna(subset=required_cols)

    # Remove extreme outliers in ReadingSpeed_wps (e.g., > 20 words/sec is very likely erroneous)
    # and extremely slow values (< 0.01 wps)
    df = df[(df['ReadingSpeed_wps'] > 0.01) & (df['ReadingSpeed_wps'] <= 20)]

    # Reset index for downstream modeling
    df = df.reset_index(drop=True)

    # Keep only columns necessary for modeling + a few intermediates for diagnostics
    keep_cols = [
        'ReaderView', 'ReadingTime_ms', 'ReadingTime_s', 'NumWords', 'ReadingSpeed_wps',
        'Dyslexia', 'DyslexiaSeverity', 'Comprehension', 'Age', 'Device', 'IsNativeEnglish',
        'Education', 'Gender', 'FKScore', 'IsRetake'
    ]
    # Ensure any columns that might be missing are created (safe-guards)
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear model (OLS) to estimate the effect of ReaderView on reading speed and test whether that effect
    differs for readers with dyslexia (interaction: ReaderView * Dyslexia).

    Returns the fitted statsmodels regression results object.
    """
    # Ensure we have the required variables; drop rows with missing values in predictors/controls used in formula
    formula = (
        'ReadingSpeed_wps ~ ReaderView * Dyslexia '
        '+ Age + NumWords + Comprehension + IsNativeEnglish + FKScore + IsRetake '
        '+ C(Device) + C(Education) + C(Gender)'
    )

    # Drop rows with NA in any variable appearing in the model
    model_vars = [
        'ReadingSpeed_wps', 'ReaderView', 'Dyslexia', 'Age', 'NumWords', 'Comprehension',
        'IsNativeEnglish', 'FKScore', 'IsRetake', 'Device', 'Education', 'Gender'
    ]
    df_model = df.dropna(subset=model_vars).copy()

    # Convert integer-like nullable dtypes to numpy-backed integer types for patsy compatibility
    # Only convert columns we know should be integer and are used in the model
    int_cols = ['ReaderView', 'Dyslexia', 'IsNativeEnglish', 'IsRetake']
    for col in int_cols:
        if col in df_model.columns:
            # safe convert since we dropped NA for model variables
            df_model[col] = df_model[col].astype('int64')

    # Ensure numeric continuous vars are numeric numpy dtypes
    num_cols = ['ReadingSpeed_wps', 'Age', 'NumWords', 'Comprehension', 'FKScore']
    for col in num_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

    # Convert categorical columns to type 'category' so statsmodels/formula handles them properly
    if 'Device' in df_model.columns:
        df_model['Device'] = df_model['Device'].astype('category')
    if 'Education' in df_model.columns:
        df_model['Education'] = df_model['Education'].astype('category')
    if 'Gender' in df_model.columns:
        # Gender is treated as categorical in the formula
        df_model['Gender'] = df_model['Gender'].astype('category')

    # Fit OLS with robust (HC3) standard errors to reduce sensitivity to heteroskedasticity
    ols_model = smf.ols(formula, data=df_model).fit(cov_type='HC3')

    # Return the fitted model object (caller can inspect ols_model.summary(), params, pvalues, etc.)
    return ols_model