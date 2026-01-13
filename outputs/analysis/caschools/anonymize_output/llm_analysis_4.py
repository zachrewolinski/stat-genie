from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.

    Produces the following new columns used in the model:
      - Enrollment: feature6 (float)
      - NumTeachers: feature7 (float), NaN if <= 0
      - StudentTeacherRatio: Enrollment / NumTeachers (winsorized at 1st/99th pct)
      - ReadingScore: feature14 (float) (with fallbacks)
      - MathScore: feature15 (float) (with fallbacks)
      - AvgScore: mean of ReadingScore and MathScore
      - PctCalWorks: feature8 (float)
      - PctReducedLunch: feature9 (float)
      - Computers: feature10 (float)
      - ComputersPerStudent: Computers / Enrollment
      - ExpenditurePerStudent: feature11 (float)
      - DistrictIncomeK: feature12 (float)
      - PctEnglishLearners: feature13 (float)
      - GradeSpan: feature5 (string)
      - County: feature4 (string)
      - LogEnrollment: log1p(Enrollment)

    Drops rows with missing values in the dependent variable (AvgScore).
    For other numeric controls and essential covariates, missing values are
    imputed with sensible medians or fallbacks to avoid dropping all observations.
    """
    df = df.copy()

    # Helper to safely get numeric series (returns NaN series if column missing)
    def numeric_series(col_name: str) -> pd.Series:
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors='coerce')
        else:
            return pd.Series(np.nan, index=df.index, dtype=float)

    # Helper to find the first candidate column that appears to contain numeric values.
    # This lets us be robust to slight variations in raw column naming.
    def first_numeric_from_candidates(candidates):
        for c in candidates:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors='coerce')
                if s.notna().any():
                    return s
        # If none of the candidates contain any non-missing numeric values,
        # return the numeric conversion of the first candidate that exists (or NaN series).
        for c in candidates:
            if c in df.columns:
                return pd.to_numeric(df[c], errors='coerce')
        # If none exist, return an all-NaN series
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Create required final columns from raw features (if raw missing, result will be NaN and later handled)
    df['Enrollment'] = numeric_series('feature6')
    df['NumTeachers'] = numeric_series('feature7')

    # Treat non-positive teacher counts as missing (we will later attempt to impute ratios)
    df['NumTeachers'] = df['NumTeachers'].where(df['NumTeachers'] > 0, np.nan)

    # Student-teacher ratio (may be NaN where NumTeachers or Enrollment missing)
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']

    # Winsorize student-teacher ratio at 1st and 99th percentiles (computed on non-missing values)
    if df['StudentTeacherRatio'].notna().any():
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        if pd.notna(lower) and pd.notna(upper):
            df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower, upper)

    # If StudentTeacherRatio is missing for some rows, impute with the median observed ratio.
    median_ratio = df['StudentTeacherRatio'].median(skipna=True)
    if pd.isna(median_ratio):
        # Try to compute a fallback median from Enrollment and NumTeachers where both present
        mask = df['Enrollment'].notna() & df['NumTeachers'].notna()
        if mask.any():
            fallback_ratios = df.loc[mask, 'Enrollment'] / df.loc[mask, 'NumTeachers']
            if not fallback_ratios.empty:
                median_ratio = fallback_ratios.median(skipna=True)
    # Final sensible fallback if still NaN (rare): use 15 students per teacher as a conservative default
    if pd.isna(median_ratio):
        median_ratio = 15.0
    df['StudentTeacherRatio'] = df['StudentTeacherRatio'].fillna(median_ratio).astype(float)

    # Scores: try to be robust to alternate raw column names for reading and math scores
    reading_candidates = ['feature14', 'reading', 'reading_score', 'read_score', 'ReadingScore', 'ReadScore']
    math_candidates = ['feature15', 'math', 'math_score', 'math_score', 'MathScore', 'Math_Score']
    df['ReadingScore'] = first_numeric_from_candidates(reading_candidates)
    df['MathScore'] = first_numeric_from_candidates(math_candidates)
    df['AvgScore'] = df[['ReadingScore', 'MathScore']].mean(axis=1)

    # Controls
    df['PctCalWorks'] = numeric_series('feature8')
    df['PctReducedLunch'] = numeric_series('feature9')
    df['Computers'] = numeric_series('feature10')
    # Avoid division by zero / missing enrollment; result will be NaN where appropriate
    df['ComputersPerStudent'] = df['Computers'] / df['Enrollment'].replace({0: np.nan})
    df['ExpenditurePerStudent'] = numeric_series('feature11')
    df['DistrictIncomeK'] = numeric_series('feature12')
    df['PctEnglishLearners'] = numeric_series('feature13')

    # Categorical controls: ensure string values and replace missing with a literal 'missing'
    if 'feature5' in df.columns:
        df['GradeSpan'] = df['feature5'].fillna('missing').astype(str)
    else:
        df['GradeSpan'] = 'missing'

    if 'feature4' in df.columns:
        df['County'] = df['feature4'].fillna('missing').astype(str)
    else:
        df['County'] = 'missing'

    # Log-transform of enrollment to reduce skew (preserves NaN)
    # Only compute where Enrollment is non-negative; negative enrollments become NaN
    df['LogEnrollment'] = np.where(df['Enrollment'] >= 0, np.log1p(df['Enrollment']), np.nan)

    # If LogEnrollment is missing for some rows, impute with median log-enrollment (or fallback from median enrollment)
    median_log = df['LogEnrollment'].median(skipna=True)
    if pd.isna(median_log):
        median_enr = df['Enrollment'].median(skipna=True)
        if pd.notna(median_enr):
            median_log = np.log1p(median_enr)
    # Final fallback if still NaN
    if pd.isna(median_log):
        median_log = 8.0  # conservative default (log1p of a large-enough enrollment)
    df['LogEnrollment'] = df['LogEnrollment'].fillna(median_log).astype(float)

    # Drop rows missing the dependent variable (AvgScore). We do not impute the dependent variable.
    # However, be robust: if AvgScore is entirely missing for the dataset (no raw score columns available),
    # we avoid dropping all rows here and instead leave the dataframe intact so the caller can decide.
    # To remain faithful to the intent, only drop rows with missing AvgScore when there is at least one non-missing AvgScore in the dataset.
    if df['AvgScore'].notna().any():
        df = df.dropna(subset=['AvgScore'])
    # Otherwise, leave rows as-is (AvgScore all NaN). The model function will handle the empty-case or missing DV.

    # For the remaining numeric control variables used by the model, impute missing values
    # with the median (or 0 if median is not available) to avoid dropping observations.
    impute_vars = [
        'ExpenditurePerStudent', 'DistrictIncomeK', 'PctReducedLunch',
        'PctEnglishLearners', 'ComputersPerStudent', 'PctCalWorks'
    ]
    for col in impute_vars:
        if col not in df.columns:
            # create column of NaNs if missing entirely
            df[col] = pd.Series(np.nan, index=df.index, dtype=float)
        median_val = df[col].median(skipna=True)
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val).astype(float)

    # Ensure GradeSpan has no missing string-like 'nan' artifacts; replace any remaining null-like strings with 'missing'
    df['GradeSpan'] = df['GradeSpan'].replace({'nan': 'missing', 'None': 'missing'}).fillna('missing').astype(str)
    df['County'] = df['County'].replace({'nan': 'missing', 'None': 'missing'}).fillna('missing').astype(str)

    # Reset index and return
    df = df.reset_index(drop=True)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and average test score.

    Model specification:
      AvgScore ~ StudentTeacherRatio + LogEnrollment + ExpenditurePerStudent + DistrictIncomeK
                 + PctReducedLunch + PctEnglishLearners + ComputersPerStudent + PctCalWorks
                 + C(GradeSpan)

    Uses robust (HC1) standard errors to guard against heteroskedasticity. Returns the fitted results object.
    """
    # Ensure a copy so we don't modify caller frame
    df_model = df.copy()

    # If there are no rows, cannot fit a model: raise a clear error
    if df_model.shape[0] == 0:
        raise ValueError("No observations available for model fitting after transform.")

    # Ensure GradeSpan has a usable categorical representation.
    # Replace any remaining null-like entries with 'missing'
    if 'GradeSpan' not in df_model.columns:
        df_model['GradeSpan'] = pd.Series(['missing'] * len(df_model), index=df_model.index)
    df_model['GradeSpan'] = df_model['GradeSpan'].replace({'nan': 'missing', 'None': 'missing'}).fillna('missing').astype(str)

    # If, after replacement, there are still zero non-missing levels (very unlikely), set all to 'missing'
    if df_model['GradeSpan'].nunique(dropna=True) == 0:
        df_model['GradeSpan'] = 'missing'

    # Convert to categorical to make the intent explicit for patsy/statsmodels
    df_model['GradeSpan'] = pd.Categorical(df_model['GradeSpan'])

    # Ensure all model columns exist. If any control columns are missing (shouldn't be after transform),
    # create them filled with zeros to avoid errors in modeling.
    expected_cols = [
        'AvgScore', 'StudentTeacherRatio', 'LogEnrollment', 'ExpenditurePerStudent', 'DistrictIncomeK',
        'PctReducedLunch', 'PctEnglishLearners', 'ComputersPerStudent', 'PctCalWorks', 'GradeSpan'
    ]
    for col in expected_cols:
        if col not in df_model.columns:
            if col == 'GradeSpan':
                df_model[col] = pd.Categorical(['missing'] * len(df_model))
            else:
                df_model[col] = 0.0

    # If AvgScore column exists but is entirely missing, fitting will fail. Raise a clear error.
    if df_model['AvgScore'].notna().sum() == 0:
        raise ValueError("Dependent variable AvgScore is missing for all observations; cannot fit model.")

    formula = (
        'AvgScore ~ StudentTeacherRatio + LogEnrollment + ExpenditurePerStudent + DistrictIncomeK '
        '+ PctReducedLunch + PctEnglishLearners + ComputersPerStudent + PctCalWorks + C(GradeSpan)'
    )

    fit = smf.ols(formula=formula, data=df_model).fit(cov_type='HC1')
    return fit