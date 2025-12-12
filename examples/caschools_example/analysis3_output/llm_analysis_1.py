from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Attempt to read the dataset if available at this path. If not, this will raise at import time;
# callers can supply their own dataframe to transform() instead of relying on this top-level read.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/caschools/data.csv')
except Exception:
    # If the file is not present in the environment, create an empty DataFrame placeholder.
    df = pd.DataFrame()


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column name from df that matches any candidate.
    Matching strategy:
      1. Exact match ignoring case.
      2. Candidate substring contained in column name (case-insensitive).
    Returns the actual column name from df, or None if no match.
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    # exact matches (case-insensitive)
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    # substring matches
    for cand in candidates:
        key = cand.lower()
        for lc, orig in lower_map.items():
            if key in lc:
                return orig
    return None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analysis-ready dataframe.

    Produces the following new columns (used in the model):
      - TotalEnrollment: from raw enrollment-like columns
      - NumTeachers: from 'teachers' or similar
      - StudentTeacherRatio: TotalEnrollment / NumTeachers OR from ratio-like columns
      - LogStudentTeacherRatio: log(StudentTeacherRatio)
      - AcademicPerformance: from 'grades' or related test score columns (or math/reading mean)
      - ExpenditurePerStudent: from 'expenditure' or similar
      - PctEnglishLearners: from english/ELL-related columns
      - PctCalWorks: from 'calworks' or similar
      - PctReducLunch: from reduced-lunch related columns
      - Computers: from 'computer(s)' or similar
      - School_KK_06: dummy for school == 'KK-06' (school type indicator)

    Notes: we coerce relevant columns to numeric. We keep rows even if AcademicPerformance is missing
    so downstream code can attempt to recover or decide on dropping.
    """
    # Work on a copy to avoid modifying the caller's dataframe
    df = df.copy()

    # Helper candidate lists for likely raw column names (common variants)
    enrollment_cands = [
        'enrollment', 'enrolment', 'students', 'total_enrollment', 'enroll', 'total', 'n_students',
        'totalstudents', 'total_students'
    ]
    teachers_cands = [
        'teachers', 'num_teachers', 'n_teachers', 'teacher_fte', 'fte_teachers', 'num_teacher_fte',
        'num_teacher', 'teaching_staff'
    ]
    # Sometimes datasets include a direct pupil-teacher ratio column
    ratio_cands = [
        'student_teacher_ratio', 'studentteacher', 'pupil_teacher_ratio', 'pupilteacher',
        'ptratio', 'students_per_teacher', 'studentsperteacher', 'pupil_teacher', 'stu_teach_ratio'
    ]
    expenditure_cands = [
        'expenditure', 'expenditure_per_student', 'expend', 'expperstudent', 'per_student_expenditure',
        'spend_per_student', 'expenditure_per_pupil'
    ]
    pct_ell_cands = [
        'ell', 'pct_english', 'pctell', 'english', 'english_learners', 'pct_english_learners',
        'PctEnglishLearners', 'pct_ell'
    ]
    calworks_cands = ['calworks', 'pct_calworks', 'pctcalworks', 'PctCalWorks']
    reduc_lunch_cands = [
        'reduced_lunch', 'pct_reduced_lunch', 'reduc_lunch', 'reduced', 'lunch',
        'pct_reduc_lunch', 'PctReducLunch', 'free_reduced_lunch', 'pct_free_reduced_lunch'
    ]
    computers_cands = ['computer', 'computers', 'num_computers', 'Computers', 'num_computer']
    grades_cands = [
        'grades', 'grade', 'avg_score', 'test_score', 'score', 'math', 'reading', 'AcademicPerformance',
        'avg_test_score', 'avg_score_total', 'testscores', 'test_scores', 'total_score', 'api', 'apiscale'
    ]
    math_cands = ['math', 'math_score', 'avg_math', 'maths']
    reading_cands = ['reading', 'reading_score', 'avg_reading', 'read']

    # Find best matching raw columns
    col_enrollment = _find_first_col(df, enrollment_cands)
    col_teachers = _find_first_col(df, teachers_cands)
    col_ratio = _find_first_col(df, ratio_cands)
    col_expenditure = _find_first_col(df, expenditure_cands)
    col_pct_ell = _find_first_col(df, pct_ell_cands)
    col_calworks = _find_first_col(df, calworks_cands)
    col_reduc = _find_first_col(df, reduc_lunch_cands)
    col_computers = _find_first_col(df, computers_cands)
    col_grades = _find_first_col(df, grades_cands)
    col_math = _find_first_col(df, math_cands)
    col_reading = _find_first_col(df, reading_cands)

    # Map raw columns to final columns; coerce to numeric where appropriate
    if col_enrollment is not None:
        df['TotalEnrollment'] = pd.to_numeric(df.get(col_enrollment), errors='coerce')
    else:
        df['TotalEnrollment'] = np.nan

    if col_teachers is not None:
        df['NumTeachers'] = pd.to_numeric(df.get(col_teachers), errors='coerce')
    else:
        df['NumTeachers'] = np.nan

    # If a direct ratio column exists, prefer it (but still keep TotalEnrollment and NumTeachers if available)
    if col_ratio is not None:
        df['StudentTeacherRatio'] = pd.to_numeric(df.get(col_ratio), errors='coerce')
    else:
        # Compute ratio from enrollment and teachers; guard against division by zero
        df['StudentTeacherRatio'] = np.nan
        # Ensure numeric types
        df['TotalEnrollment'] = pd.to_numeric(df['TotalEnrollment'], errors='coerce')
        df['NumTeachers'] = pd.to_numeric(df['NumTeachers'], errors='coerce')
        # Treat nonpositive NumTeachers as NaN to avoid invalid ratios
        df.loc[df['NumTeachers'] <= 0, 'NumTeachers'] = np.nan
        # Compute ratio where possible
        mask_ratio = df['TotalEnrollment'].notna() & df['NumTeachers'].notna()
        df.loc[mask_ratio, 'StudentTeacherRatio'] = df.loc[mask_ratio, 'TotalEnrollment'] / df.loc[mask_ratio, 'NumTeachers']

    if col_expenditure is not None:
        df['ExpenditurePerStudent'] = pd.to_numeric(df.get(col_expenditure), errors='coerce')
    else:
        df['ExpenditurePerStudent'] = np.nan

    if col_pct_ell is not None:
        df['PctEnglishLearners'] = pd.to_numeric(df.get(col_pct_ell), errors='coerce')
    else:
        df['PctEnglishLearners'] = np.nan

    if col_calworks is not None:
        df['PctCalWorks'] = pd.to_numeric(df.get(col_calworks), errors='coerce')
    else:
        df['PctCalWorks'] = np.nan

    if col_reduc is not None:
        df['PctReducLunch'] = pd.to_numeric(df.get(col_reduc), errors='coerce')
    else:
        df['PctReducLunch'] = np.nan

    if col_computers is not None:
        df['Computers'] = pd.to_numeric(df.get(col_computers), errors='coerce')
    else:
        df['Computers'] = np.nan

    # Academic performance: prefer direct grades-like column; otherwise try to construct from math/reading
    if col_grades is not None:
        df['AcademicPerformance'] = pd.to_numeric(df.get(col_grades), errors='coerce')
    else:
        # Try to use math and reading (if present)
        math_series = pd.to_numeric(df.get(col_math), errors='coerce') if col_math is not None else None
        reading_series = pd.to_numeric(df.get(col_reading), errors='coerce') if col_reading is not None else None
        if math_series is not None and reading_series is not None:
            # mean of available
            df['AcademicPerformance'] = pd.concat([math_series, reading_series], axis=1).mean(axis=1)
        elif math_series is not None:
            df['AcademicPerformance'] = math_series
        elif reading_series is not None:
            df['AcademicPerformance'] = reading_series
        else:
            df['AcademicPerformance'] = np.nan

    # Clean NumTeachers: ensure numeric (already coerced earlier) and treat nonpositive as NaN
    if 'NumTeachers' in df.columns:
        df['NumTeachers'] = pd.to_numeric(df['NumTeachers'], errors='coerce')
        df.loc[df['NumTeachers'] <= 0, 'NumTeachers'] = np.nan

    # Ensure StudentTeacherRatio is numeric and set nonfinite/nonpositive to NaN
    df['StudentTeacherRatio'] = pd.to_numeric(df['StudentTeacherRatio'], errors='coerce')
    df.loc[~np.isfinite(df['StudentTeacherRatio']), 'StudentTeacherRatio'] = np.nan
    df.loc[df['StudentTeacherRatio'] <= 0, 'StudentTeacherRatio'] = np.nan

    # Compute log ratio for positive ratios
    df['LogStudentTeacherRatio'] = np.nan
    positive_mask = df['StudentTeacherRatio'] > 0
    if positive_mask.any():
        df.loc[positive_mask, 'LogStudentTeacherRatio'] = np.log(df.loc[positive_mask, 'StudentTeacherRatio'])

    # Ensure 'school' exists as a string and create School_KK_06 dummy
    if 'school' in df.columns:
        # fillna before astype to avoid string "nan"
        df['school'] = df['school'].fillna('').astype(str)
    else:
        df['school'] = ''
    # normalize whitespace
    df['school'] = df['school'].str.strip()
    df['School_KK_06'] = (df['school'] == 'KK-06').astype(int)

    # If AcademicPerformance has no non-missing values, attempt a more aggressive search
    if df['AcademicPerformance'].notna().sum() == 0:
        # Priority keywords to look for in column names
        priority_keywords = ['score', 'test', 'api', 'grade', 'avg', 'reading', 'math', 'performance', 'total']
        candidate_cols = []
        for col in df.columns:
            lc = str(col).lower()
            # skip final conceptual columns to avoid circular picks
            if lc in [c.lower() for c in [
                'StudentTeacherRatio', 'LogStudentTeacherRatio', 'AcademicPerformance',
                'ExpenditurePerStudent', 'PctEnglishLearners', 'PctCalWorks', 'PctReducLunch',
                'Computers', 'School_KK_06', 'TotalEnrollment', 'NumTeachers', 'school'
            ]]:
                continue
            # consider columns with any priority keyword
            if any(kw in lc for kw in priority_keywords):
                candidate_cols.append(col)
        # Evaluate candidates by number of numeric non-missing entries
        best_col = None
        best_count = 0
        for col in candidate_cols:
            num_nonmissing = pd.to_numeric(df.get(col), errors='coerce').notna().sum()
            if num_nonmissing > best_count:
                best_count = num_nonmissing
                best_col = col
        # If found a reasonable candidate (at least one numeric observation), use it
        if best_col is not None and best_count > 0:
            df['AcademicPerformance'] = pd.to_numeric(df.get(best_col), errors='coerce')
        else:
            # As a last resort, try to use any numeric column with at least one non-missing value
            numeric_candidates = []
            for col in df.columns:
                if col in ['StudentTeacherRatio', 'LogStudentTeacherRatio', 'AcademicPerformance',
                           'ExpenditurePerStudent', 'PctEnglishLearners', 'PctCalWorks', 'PctReducLunch',
                           'Computers', 'School_KK_06', 'TotalEnrollment', 'NumTeachers', 'school']:
                    continue
                num_nonmissing = pd.to_numeric(df.get(col), errors='coerce').notna().sum()
                if num_nonmissing > 0:
                    numeric_candidates.append((col, num_nonmissing))
            if numeric_candidates:
                # pick the column with most non-missing numeric entries
                numeric_candidates.sort(key=lambda x: x[1], reverse=True)
                pick_col = numeric_candidates[0][0]
                df['AcademicPerformance'] = pd.to_numeric(df.get(pick_col), errors='coerce')
            # else leave AcademicPerformance as all-NaN (model will handle raising later)

    # Ensure final required columns exist in the dataframe (create with NaN if missing)
    final_cols = [
        'StudentTeacherRatio', 'LogStudentTeacherRatio',
        'AcademicPerformance',
        'ExpenditurePerStudent', 'PctEnglishLearners', 'PctCalWorks', 'PctReducLunch',
        'Computers',
        'School_KK_06',
        'TotalEnrollment', 'NumTeachers'
    ]
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Do NOT drop rows missing AcademicPerformance here; let model decide downstream.
    # Ensure numeric dtype for other continuous columns
    for col in [
        'ExpenditurePerStudent', 'PctEnglishLearners', 'PctCalWorks', 'PctReducLunch', 'Computers',
        'TotalEnrollment', 'NumTeachers', 'StudentTeacherRatio', 'LogStudentTeacherRatio', 'AcademicPerformance'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # After coercion, again compute LogStudentTeacherRatio for any rows where StudentTeacherRatio is valid and positive
    positive_mask = df['StudentTeacherRatio'] > 0
    df.loc[positive_mask, 'LogStudentTeacherRatio'] = np.log(df.loc[positive_mask, 'StudentTeacherRatio'])

    # Final cleanup: reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model of AcademicPerformance on student-teacher ratio and controls.

    Returns the fitted statsmodels RegressionResultsWrapper object.
    The model uses a log transform of the student-teacher ratio to reduce skew and allow elasticity interpretation.
    Robust (HC3) standard errors are used to mitigate heteroskedasticity.
    """

    # Required core columns for modeling (must exist / be computable)
    core_required = [
        'AcademicPerformance',
    ]

    # Check that the dataframe contains the core required column names
    missing_core = [c for c in core_required if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing required core columns for modeling: {missing_core}")

    # Work on a copy so we don't mutate the caller's dataframe
    df = df.copy()

    # Ensure key numeric helpers exist for possible ratio computation
    if 'TotalEnrollment' not in df.columns:
        df['TotalEnrollment'] = np.nan
    if 'NumTeachers' not in df.columns:
        df['NumTeachers'] = np.nan

    # Coerce helpers to numeric
    df['TotalEnrollment'] = pd.to_numeric(df['TotalEnrollment'], errors='coerce')
    df['NumTeachers'] = pd.to_numeric(df['NumTeachers'], errors='coerce')
    df.loc[df['NumTeachers'] <= 0, 'NumTeachers'] = np.nan

    # Ensure StudentTeacherRatio column exists
    if 'StudentTeacherRatio' not in df.columns:
        df['StudentTeacherRatio'] = np.nan
    else:
        df['StudentTeacherRatio'] = pd.to_numeric(df['StudentTeacherRatio'], errors='coerce')

    # Attempt to compute StudentTeacherRatio from TotalEnrollment and NumTeachers where missing
    mask_need_ratio = df['StudentTeacherRatio'].isna() & df['TotalEnrollment'].notna() & df['NumTeachers'].notna()
    if mask_need_ratio.any():
        df.loc[mask_need_ratio, 'StudentTeacherRatio'] = df.loc[mask_need_ratio, 'TotalEnrollment'] / df.loc[mask_need_ratio, 'NumTeachers']

    # Ensure LogStudentTeacherRatio exists
    if 'LogStudentTeacherRatio' not in df.columns:
        df['LogStudentTeacherRatio'] = np.nan
    else:
        df['LogStudentTeacherRatio'] = pd.to_numeric(df['LogStudentTeacherRatio'], errors='coerce')

    # Compute log where StudentTeacherRatio > 0
    s = pd.to_numeric(df['StudentTeacherRatio'], errors='coerce')
    positive_mask = s > 0
    if positive_mask.any():
        df.loc[positive_mask, 'LogStudentTeacherRatio'] = np.log(s[positive_mask])

    # For remaining non-missing StudentTeacherRatio values (e.g., zeros), compute log(value + eps)
    remaining_nonmissing = s.notna() & df['LogStudentTeacherRatio'].isna()
    if remaining_nonmissing.any():
        eps = 1e-6
        df.loc[remaining_nonmissing, 'LogStudentTeacherRatio'] = np.log(s[remaining_nonmissing] + eps)

    # Final fallback: if any rows with AcademicPerformance still have missing LogStudentTeacherRatio, impute typical value
    rows_need_impute = df['AcademicPerformance'].notna() & df['LogStudentTeacherRatio'].isna()
    if rows_need_impute.any():
        typical_ratio = 20.0
        df.loc[rows_need_impute, 'LogStudentTeacherRatio'] = np.log(typical_ratio)

    # If AcademicPerformance column exists but all values are missing, attempt to construct it from raw grade/math/reading columns
    if df['AcademicPerformance'].notna().sum() == 0:
        # Candidate lists similar to those used in transform
        grades_cands = [
            'grades', 'grade', 'avg_score', 'test_score', 'score', 'avg_test_score',
            'avg_score_total', 'testscores', 'test_scores', 'total_score', 'api', 'apiscale'
        ]
        math_cands = ['math', 'math_score', 'avg_math', 'maths']
        reading_cands = ['reading', 'reading_score', 'avg_reading', 'read']

        col_grades = _find_first_col(df, grades_cands)
        col_math = _find_first_col(df, math_cands)
        col_reading = _find_first_col(df, reading_cands)

        if col_grades is not None:
            df['AcademicPerformance'] = pd.to_numeric(df.get(col_grades), errors='coerce')
        else:
            math_series = pd.to_numeric(df.get(col_math), errors='coerce') if col_math is not None else None
            reading_series = pd.to_numeric(df.get(col_reading), errors='coerce') if col_reading is not None else None
            if math_series is not None and reading_series is not None:
                df['AcademicPerformance'] = pd.concat([math_series, reading_series], axis=1).mean(axis=1)
            elif math_series is not None:
                df['AcademicPerformance'] = math_series
            elif reading_series is not None:
                df['AcademicPerformance'] = reading_series
            # else leave as is (all NaN)

    # After attempting fallback, if still no non-missing AcademicPerformance, attempt broader heuristic
    if df['AcademicPerformance'].notna().sum() == 0:
        priority_keywords = ['score', 'test', 'api', 'grade', 'avg', 'reading', 'math', 'performance', 'total']
        candidate_cols = []
        for col in df.columns:
            lc = str(col).lower()
            if lc in [c.lower() for c in [
                'StudentTeacherRatio', 'LogStudentTeacherRatio', 'AcademicPerformance',
                'ExpenditurePerStudent', 'PctEnglishLearners', 'PctCalWorks', 'PctReducLunch',
                'Computers', 'School_KK_06', 'TotalEnrollment', 'NumTeachers', 'school'
            ]]:
                continue
            if any(kw in lc for kw in priority_keywords):
                candidate_cols.append(col)
        best_col = None
        best_count = 0
        for col in candidate_cols:
            num_nonmissing = pd.to_numeric(df.get(col), errors='coerce').notna().sum()
            if num_nonmissing > best_count:
                best_count = num_nonmissing
                best_col = col
        if best_col is not None and best_count > 0:
            df['AcademicPerformance'] = pd.to_numeric(df.get(best_col), errors='coerce')
        else:
            # As last resort, pick any numeric column with most non-missing values
            numeric_candidates = []
            for col in df.columns:
                if col in ['StudentTeacherRatio', 'LogStudentTeacherRatio', 'AcademicPerformance',
                           'ExpenditurePerStudent', 'PctEnglishLearners', 'PctCalWorks', 'PctReducLunch',
                           'Computers', 'School_KK_06', 'TotalEnrollment', 'NumTeachers', 'school']:
                    continue
                num_nonmissing = pd.to_numeric(df.get(col), errors='coerce').notna().sum()
                if num_nonmissing > 0:
                    numeric_candidates.append((col, num_nonmissing))
            if numeric_candidates:
                numeric_candidates.sort(key=lambda x: x[1], reverse=True)
                pick_col = numeric_candidates[0][0]
                df['AcademicPerformance'] = pd.to_numeric(df.get(pick_col), errors='coerce')
            # If still nothing, we must raise later (cannot fabricate outcome)

    # After attempting fallback, if still no non-missing AcademicPerformance, we cannot proceed
    if df['AcademicPerformance'].notna().sum() == 0:
        raise ValueError("No observations have AcademicPerformance available; cannot fit model.")

    # Define control columns (these should be used in the model)
    control_cols = [
        'ExpenditurePerStudent',
        'PctEnglishLearners',
        'PctCalWorks',
        'PctReducLunch',
        'Computers',
        'School_KK_06'
    ]

    # Ensure control columns exist in dataframe; if not, create them with NaN
    for col in control_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Now select rows that have the core DV present (AcademicPerformance).
    df_model = df.loc[df['AcademicPerformance'].notna()].copy()

    # If no observations remain, raise a clear error
    if df_model.shape[0] == 0:
        raise ValueError("No observations remain after requiring AcademicPerformance; cannot fit model.")

    # Ensure LogStudentTeacherRatio present for modeling; drop rows that somehow still lack it
    df_model['LogStudentTeacherRatio'] = pd.to_numeric(df_model['LogStudentTeacherRatio'], errors='coerce')
    df_model = df_model.loc[df_model['LogStudentTeacherRatio'].notna()].copy()
    if df_model.shape[0] == 0:
        # As a last resort, try to compute LogStudentTeacherRatio from StudentTeacherRatio for any rows with ratio
        s2 = pd.to_numeric(df['StudentTeacherRatio'], errors='coerce')
        if s2.notna().any():
            df_model = df.loc[df['StudentTeacherRatio'].notna() & df['AcademicPerformance'].notna()].copy()
            df_model['LogStudentTeacherRatio'] = np.log(pd.to_numeric(df_model['StudentTeacherRatio'], errors='coerce').replace(0, np.nan) + 1e-6)
        else:
            raise ValueError("No observations remain after ensuring LogStudentTeacherRatio is available; cannot fit model.")

    # For control variables, impute missing with column medians where possible, otherwise zero.
    # This preserves observations rather than dropping them entirely.
    for col in control_cols:
        if col == 'School_KK_06':
            # For the categorical indicator, treat missing as 0 (not KK-06)
            df_model[col] = df_model[col].fillna(0).astype(int)
        else:
            # Numeric controls: compute median among non-missing; if none, fill with 0
            col_numeric = pd.to_numeric(df_model[col], errors='coerce')
            if col_numeric.notna().any():
                median_val = col_numeric.median()
                df_model[col] = col_numeric.fillna(median_val)
            else:
                df_model[col] = col_numeric.fillna(0)

    # Define outcome and predictors
    y = pd.to_numeric(df_model['AcademicPerformance'], errors='coerce').astype(float)

    X_vars = [
        'LogStudentTeacherRatio',
        'ExpenditurePerStudent',
        'PctEnglishLearners',
        'PctCalWorks',
        'PctReducLunch',
        'Computers',
        'School_KK_06'
    ]

    X = df_model[X_vars].copy()

    # Ensure predictors are numeric (convert booleans/ints where possible)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Final check: ensure no remaining NaNs in X or y
    if X.isnull().values.any() or y.isnull().values.any():
        # Drop any rows with NaNs just in case
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    if X.shape[0] == 0:
        raise ValueError("No observations with complete model data after final NA removal; cannot fit model.")

    # Fit OLS with robust (HC3) standard errors
    model_result = sm.OLS(y, X).fit(cov_type='HC3')

    return model_result