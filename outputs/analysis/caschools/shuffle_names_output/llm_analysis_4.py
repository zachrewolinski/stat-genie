from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw dataframe into the FINAL dataframe required by the model.

    Required output columns (must be present and spelled exactly as below):
      - StudentTeacherRatio
      - AcademicPerformance
      - TotalEnrollment
      - NumTeachers
      - ExpenditurePerStudent
      - PctReducedLunch
      - PctEnglishLearners
      - log_STR
      - log_Expenditure
      - analytic_sample

    The function attempts to read the following raw columns (if present) and map
    them to the conceptual variables:
      - 'calworks' -> TotalEnrollment
      - 'teachers' -> NumTeachers
      - 'grades' -> AcademicPerformance
      - 'expenditure' -> ExpenditurePerStudent
      - 'math' -> PctReducedLunch
      - 'district' -> PctEnglishLearners

    If any of the raw columns are missing they are treated as all-NaN.
    """
    df = df.copy()

    def _to_numeric_or_na(src_df: pd.DataFrame, col_name: str) -> pd.Series:
        if col_name in src_df.columns:
            return pd.to_numeric(src_df[col_name], errors='coerce')
        else:
            return pd.Series(np.nan, index=src_df.index)

    # Map and convert to numeric (coerce errors to NaN)
    df['TotalEnrollment'] = _to_numeric_or_na(df, 'calworks')
    df['NumTeachers'] = _to_numeric_or_na(df, 'teachers')
    df['AcademicPerformance'] = _to_numeric_or_na(df, 'grades')
    df['ExpenditurePerStudent'] = _to_numeric_or_na(df, 'expenditure')
    df['PctReducedLunch'] = _to_numeric_or_na(df, 'math')
    df['PctEnglishLearners'] = _to_numeric_or_na(df, 'district')

    # Basic data cleaning: remove observations with missing core variables or invalid values
    # Need enrollment, teachers, and outcome
    df = df.dropna(subset=['TotalEnrollment', 'NumTeachers', 'AcademicPerformance'])

    # Remove observations with nonpositive teachers to avoid division by zero or implausible values
    df = df[df['NumTeachers'] > 0]

    # Remove observations with extremely small or zero enrollment (not meaningful for ratio-based analysis)
    df = df[df['TotalEnrollment'] >= 1]

    # Compute the student-teacher ratio
    df['StudentTeacherRatio'] = df['TotalEnrollment'] / df['NumTeachers']

    # Create log-transformed variants for potential nonlinearity
    # Use log1p to avoid log(0). For expenditure, fillna with 0 prior to log1p.
    df['log_STR'] = np.log1p(df['StudentTeacherRatio'])
    df['log_Expenditure'] = np.log1p(df['ExpenditurePerStudent'].fillna(0))

    # Keep control columns (they may contain NaNs; model will handle with listwise deletion)
    # For transparency, create analytic sample indicator (rows with no missing values among required model vars)
    required_cols = [
        'StudentTeacherRatio',
        'AcademicPerformance',
        'ExpenditurePerStudent',
        'PctReducedLunch',
        'PctEnglishLearners',
        'TotalEnrollment',
        'NumTeachers',
        'log_STR',
        'log_Expenditure',
    ]
    df['analytic_sample'] = ~df[required_cols].isnull().any(axis=1)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit statistical models on the FINAL dataframe produced by transform().

    Returns a dictionary with:
      - 'ols_model': OLS fit of AcademicPerformance on StudentTeacherRatio and controls (HC3 SEs) or None if not estimable
      - 'ols_model_log': OLS fit using log_STR and log_Expenditure (HC3 SEs) or None if not estimable
      - 'n_obs': number of observations used in the models (analytic sample)
    """
    # Ensure we operate on a copy
    df = df.copy()

    # Subset to analytic sample (complete cases for modeling variables)
    if 'analytic_sample' not in df.columns:
        # If analytic_sample missing, fall back to constructing it conservatively
        required_cols = [
            'StudentTeacherRatio',
            'AcademicPerformance',
            'ExpenditurePerStudent',
            'PctReducedLunch',
            'PctEnglishLearners',
            'TotalEnrollment',
            'NumTeachers',
            'log_STR',
            'log_Expenditure',
        ]
        df['analytic_sample'] = ~df[required_cols].isnull().any(axis=1)

    model_df = df[df['analytic_sample']].copy()

    n_obs = int(model_df.shape[0])

    # If there are no observations in the analytic sample, avoid calling statsmodels on empty arrays
    if n_obs == 0:
        return {
            'ols_model': None,
            'ols_model_log': None,
            'n_obs': 0,
        }

    # Define dependent and independent variables for level specification
    y = model_df['AcademicPerformance']
    X = model_df[['StudentTeacherRatio', 'ExpenditurePerStudent', 'PctReducedLunch', 'PctEnglishLearners']]
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust standard errors (HC3)
    ols_model = sm.OLS(y, X).fit(cov_type='HC3')

    # Secondary specification: log functional form for STR and Expenditure
    X_log = model_df[['log_STR', 'log_Expenditure', 'PctReducedLunch', 'PctEnglishLearners']]
    X_log = sm.add_constant(X_log, has_constant='add')
    ols_model_log = sm.OLS(y, X_log).fit(cov_type='HC3')

    results = {
        'ols_model': ols_model,
        'ols_model_log': ols_model_log,
        'n_obs': n_obs,
    }

    return results