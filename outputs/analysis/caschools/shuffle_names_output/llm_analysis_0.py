from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe to create the variables needed for modeling.

    Produces a dataframe with the required final columns:
      - 'AvgTestScore'
      - 'STR_z'
      - 'Expenditure_z'
      - 'Lunch_z'
      - 'EnglishLearners_z'
      - 'Computers_z'
      - 'Income_z'
    """
    df = df.copy()

    # Ensure numeric columns are numeric where appropriate.
    raw_numeric = [
        'calworks', 'teachers', 'grades', 'expenditure', 'math', 'district',
        'english', 'income', 'enrollment', 'students', 'enroll', 'total_students', 'total_enrollment'
    ]
    for c in raw_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Dependent variable: average test score (use raw 'grades')
    # Do not drop rows with missing grades; instead impute missing AvgTestScore with column mean (or zero if mean unavailable)
    if 'grades' in df.columns:
        avg_scores = pd.to_numeric(df['grades'], errors='coerce')
    else:
        avg_scores = pd.Series(np.nan, index=df.index)

    avg_mean = avg_scores.mean(skipna=True)
    if np.isnan(avg_mean):
        avg_filled = avg_scores.fillna(0.0)
    else:
        avg_filled = avg_scores.fillna(avg_mean)
    df['AvgTestScore'] = avg_filled

    # Determine an enrollment column to compute student-teacher ratio.
    enrollment_candidates = [
        'calworks', 'enrollment', 'students', 'enroll', 'total_students', 'total_enrollment'
    ]
    enrollment_series = None
    for col in enrollment_candidates:
        if col in df.columns:
            enrollment_series = pd.to_numeric(df[col], errors='coerce')
            break
    if enrollment_series is None:
        # No enrollment-like column present; create an all-NaN series to be handled below.
        enrollment_series = pd.Series(np.nan, index=df.index)

    # Impute enrollment missing values with the column mean if available; otherwise fill with zero
    enroll_mean = enrollment_series.mean(skipna=True)
    if np.isnan(enroll_mean):
        enroll_filled = enrollment_series.fillna(0.0)
    else:
        enroll_filled = enrollment_series.fillna(enroll_mean)

    # Determine teacher counts column (try several candidate names)
    teacher_candidates = ['teachers', 'teaching_staff', 'num_teachers', 'fte_teachers', 'staff']
    teacher_series = None
    for col in teacher_candidates:
        if col in df.columns:
            teacher_series = pd.to_numeric(df[col], errors='coerce')
            break
    if teacher_series is None:
        # No teacher-like column present; create an all-NaN series
        teacher_series = pd.Series(np.nan, index=df.index)

    # Impute teacher missing values with the column mean if available; otherwise keep as NaN.
    teacher_mean = teacher_series.mean(skipna=True)
    if np.isnan(teacher_mean):
        teacher_filled = teacher_series.copy()
    else:
        teacher_filled = teacher_series.fillna(teacher_mean)

    # Avoid nonpositive teacher counts: set nonpositive values to NaN to prevent invalid ratios
    teacher_filled_nonpos = teacher_filled.where(teacher_filled > 0, other=np.nan)

    # Independent variable: student-teacher ratio (students per teacher)
    # Use the imputed enrollment series divided by teachers (teachers may be NaN -> ratio NaN)
    # Ensure alignment by index
    student_teacher_ratio = enroll_filled.reindex(df.index) / teacher_filled_nonpos.reindex(df.index)
    df['StudentTeacherRatio'] = student_teacher_ratio

    # Controls mapping from raw columns (use column names as specified in prompt)
    df['ExpenditurePerStudent'] = df['expenditure'] if 'expenditure' in df.columns else np.nan
    df['PercentReducedLunch'] = df['math'] if 'math' in df.columns else np.nan
    df['PercentEnglishLearners'] = df['district'] if 'district' in df.columns else np.nan
    df['NumComputers'] = df['english'] if 'english' in df.columns else np.nan
    df['DistrictIncome'] = df['income'] if 'income' in df.columns else np.nan

    # Helper: function to compute z-score robustly, handling constant columns and all-NaN columns
    def zscore_filled(series: pd.Series) -> pd.Series:
        # Ensure numeric
        series = pd.to_numeric(series, errors='coerce')
        mean = series.mean(skipna=True)
        if np.isnan(mean):
            # If entire column is NaN, return zeros
            return pd.Series(0.0, index=series.index)
        # Fill missing values with the column mean (imputation to avoid dropping rows)
        filled = series.fillna(mean)
        std = filled.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=series.index)
        return (filled - mean) / std

    # Compute z-scores for the independent variable and controls
    df['STR_z'] = zscore_filled(df['StudentTeacherRatio'])
    df['Expenditure_z'] = zscore_filled(df['ExpenditurePerStudent'])
    df['Lunch_z'] = zscore_filled(df['PercentReducedLunch'])
    df['EnglishLearners_z'] = zscore_filled(df['PercentEnglishLearners'])
    df['Computers_z'] = zscore_filled(df['NumComputers'])
    df['Income_z'] = zscore_filled(df['DistrictIncome'])

    # Assemble final dataframe with exactly the required final columns (may include helper columns internally)
    final_columns = [
        'AvgTestScore',
        'STR_z',
        'Expenditure_z',
        'Lunch_z',
        'EnglishLearners_z',
        'Computers_z',
        'Income_z'
    ]

    # Ensure all final columns exist in df
    for col in final_columns:
        if col not in df.columns:
            # if something unexpected happened, create a column of zeros (safe fallback)
            df[col] = 0.0

    final_df = df.loc[:, final_columns].reset_index(drop=True)

    return final_df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and academic performance.

    Model specification:
      AvgTestScore_i = beta0 + beta1 * STR_z_i + beta2 * Expenditure_z_i + beta3 * Lunch_z_i
                       + beta4 * EnglishLearners_z_i + beta5 * Computers_z_i + beta6 * Income_z_i + eps_i

    - Uses heteroskedasticity-robust standard errors (HC3).
    - Returns the fitted statsmodels regression results object.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['AvgTestScore', 'STR_z', 'Expenditure_z', 'Lunch_z', 'EnglishLearners_z', 'Computers_z', 'Income_z']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for model: {missing}")

    # Drop rows with missing outcome or predictors (should be none if transform filled appropriately)
    df = df.dropna(subset=required)
    if df.shape[0] == 0:
        raise ValueError("No observations available for modeling after transform.")

    # Build design matrix
    X = df[['STR_z', 'Expenditure_z', 'Lunch_z', 'EnglishLearners_z', 'Computers_z', 'Income_z']].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = pd.to_numeric(df['AvgTestScore'], errors='coerce')

    # Drop any remaining rows where y is missing
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    if X.shape[0] == 0:
        raise ValueError("No valid observations with non-missing outcome for modeling.")

    # Fit OLS with robust (HC3) standard errors
    model_res = sm.OLS(y, X).fit(cov_type='HC3')

    return model_res