from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _find_column(df: pd.DataFrame, candidates):
    """
    Helper to find the first column in df that matches any of the candidate names.
    Matching is case-insensitive and ignores surrounding whitespace.
    Returns the actual column name from df or None if not found.
    """
    cols_map = {c.lower().strip(): c for c in df.columns}
    # direct lower/strip match
    for cand in candidates:
        key = str(cand).lower().strip()
        if key in cols_map:
            return cols_map[key]
    # try fuzzy match by removing non-alphanumeric characters
    def normalize(s):
        return re.sub(r"\W+", "", str(s).lower().strip())

    norm_cols = {normalize(c): c for c in df.columns}
    for cand in candidates:
        nc = normalize(cand)
        if nc in norm_cols:
            return norm_cols[nc]
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the columns required for the statistical model.

    Output columns (exact names required by downstream code):
    - StudentTeacherRatio
    - AvgScore
    - ExpPerStudent
    - PctReducedLunch
    - PctCalWorks
    - PctEngLearners
    - AvgIncomeK
    - ComputersPerStudent
    - GradeSpan_KK08
    - County
    """
    # Work on a copy to avoid mutating the original
    df = df.copy()

    # Candidate names for the raw inputs. Include the original 'featureX' names
    # as well as common alternative names to make the transformer robust.
    candidates = {
        'feature6': ['feature6', 'enroll', 'enrollment', 'students', 'num_students', 'student_count'],
        'feature7': ['feature7', 'teachers', 'num_teachers', 'fte_teachers', 'fte', 'teacher_count'],
        'feature14': ['feature14', 'reading', 'read', 'avg_read', 'avg_reading', 'reading_score'],
        'feature15': ['feature15', 'math', 'avg_math', 'avg_maths', 'math_score'],
        'feature11': ['feature11', 'expenditure', 'exp_per_student', 'exp_per_stu', 'expend', 'expenditure_per_student'],
        'feature9': [
            'feature9', 'pct_reduced_lunch', 'reduced_lunch', 'frl', 'pctfrl',
            'percent_reduced_lunch', 'meal', 'lunch', 'free_lunch', 'lunch_pct'
        ],
        'feature8': ['feature8', 'pct_calworks', 'calworks', 'calworks_pct'],
        'feature13': [
            'feature13', 'pct_eng_learners', 'eng_learners', 'ell', 'pct_ell',
            'english_learners', 'pct_english_learners', 'english', 'eng', 'pct_english'
        ],
        'feature12': ['feature12', 'avginc', 'avg_income', 'avg_income_k', 'avg_income_thousands', 'income'],
        'feature10': ['feature10', 'computers', 'num_computers', 'computer', 'computers_count'],
        'feature5': [
            'feature5', 'gradespan', 'grade_span', 'gradeSpan', 'grade_span_code',
            'grades', 'grade', 'grade_span_code', 'grade_spans'
        ],
        'feature4': ['feature4', 'county', 'County', 'county_name', 'cnty']
    }

    # Map each required raw feature to an actual column present in df
    mapped = {}
    missing = []
    for key, cand_list in candidates.items():
        col = _find_column(df, cand_list)
        if col is None:
            missing.append((key, cand_list))
        else:
            mapped[key] = col

    if missing:
        # Provide a helpful error message listing available columns
        avail = list(df.columns)
        missing_keys = [k for k, _ in missing]
        raise KeyError(
            f"Required raw columns not found for keys: {missing_keys}. "
            f"Tried candidate names: {[c for _, c in missing]}. "
            f"Available columns: {avail}"
        )

    # Create clean, numeric helper columns from the mapped raw columns
    # Convert to numeric where appropriate (coerce errors to NaN)
    df['_enroll'] = pd.to_numeric(df[mapped['feature6']], errors='coerce')
    df['_teachers'] = pd.to_numeric(df[mapped['feature7']], errors='coerce')
    df['_reading'] = pd.to_numeric(df[mapped['feature14']], errors='coerce')
    df['_math'] = pd.to_numeric(df[mapped['feature15']], errors='coerce')
    df['_exp_per_student'] = pd.to_numeric(df[mapped['feature11']], errors='coerce')
    df['_pct_reduced_lunch'] = pd.to_numeric(df[mapped['feature9']], errors='coerce')
    df['_pct_calworks'] = pd.to_numeric(df[mapped['feature8']], errors='coerce')
    df['_pct_eng_learners'] = pd.to_numeric(df[mapped['feature13']], errors='coerce')
    df['_avg_income_k'] = pd.to_numeric(df[mapped['feature12']], errors='coerce')
    df['_computers'] = pd.to_numeric(df[mapped['feature10']], errors='coerce')
    df['_gradespan_raw'] = df[mapped['feature5']].astype(str)
    df['_county_raw'] = df[mapped['feature4']].astype(str)

    # Drop rows missing any required raw inputs (after coercion to numeric)
    helper_required = [
        '_enroll', '_teachers', '_reading', '_math', '_exp_per_student',
        '_pct_reduced_lunch', '_pct_calworks', '_pct_eng_learners',
        '_avg_income_k', '_computers', '_gradespan_raw', '_county_raw'
    ]
    df = df.dropna(subset=helper_required).reset_index(drop=True)

    # Avoid division by zero for teacher counts
    df.loc[df['_teachers'] == 0, '_teachers'] = np.nan
    df = df.dropna(subset=['_teachers']).reset_index(drop=True)

    # Dependent variable: average of reading and math
    df['AvgScore'] = df[['_reading', '_math']].mean(axis=1)

    # Independent variable: student-teacher ratio = enrollment / teachers
    df['StudentTeacherRatio'] = df['_enroll'] / df['_teachers']

    # Winsorize StudentTeacherRatio to reduce influence of extreme outliers (1st and 99th percentiles)
    if not df['StudentTeacherRatio'].empty:
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=lower, upper=upper)

    # Controls and auxiliary variables (final required column names)
    df['ExpPerStudent'] = df['_exp_per_student']
    df['PctReducedLunch'] = df['_pct_reduced_lunch']
    df['PctCalWorks'] = df['_pct_calworks']
    df['PctEngLearners'] = df['_pct_eng_learners']
    df['AvgIncomeK'] = df['_avg_income_k']

    # Computers per student - avoid division by zero for enrollment
    df.loc[df['_enroll'] == 0, '_enroll'] = np.nan
    df['ComputersPerStudent'] = df['_computers'] / df['_enroll']

    # Grade span indicator: 1 if 'KK-08', else 0 (tolerant to whitespace/case)
    df['GradeSpan_KK08'] = df['_gradespan_raw'].apply(lambda x: 1 if str(x).strip().upper() == 'KK-08' else 0)

    # County (categorical) - keep as string for formula C(County)
    df['County'] = df['_county_raw'].astype(str)

    # Keep only the columns needed for modeling to avoid accidental use of raw features
    model_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'ExpPerStudent',
        'PctReducedLunch',
        'PctCalWorks',
        'PctEngLearners',
        'AvgIncomeK',
        'ComputersPerStudent',
        'GradeSpan_KK08',
        'County'
    ]

    final_df = df[model_cols].reset_index(drop=True)
    return final_df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for district covariates.

    Formula:
      AvgScore ~ StudentTeacherRatio + ExpPerStudent + PctReducedLunch + PctCalWorks
                 + PctEngLearners + AvgIncomeK + ComputersPerStudent + GradeSpan_KK08
                 + C(County)

    Robust (HC3) standard errors are used.
    Returns the fitted statsmodels results object with robust covariance.
    """
    # Drop any remaining missing values in model columns
    df_model = df.dropna(subset=[
        'AvgScore', 'StudentTeacherRatio', 'ExpPerStudent', 'PctReducedLunch',
        'PctCalWorks', 'PctEngLearners', 'AvgIncomeK', 'ComputersPerStudent', 'GradeSpan_KK08', 'County'
    ]).reset_index(drop=True)

    formula = (
        'AvgScore ~ StudentTeacherRatio + ExpPerStudent + PctReducedLunch + PctCalWorks '
        '+ PctEngLearners + AvgIncomeK + ComputersPerStudent + GradeSpan_KK08 + C(County)'
    )

    # Fit the OLS model
    results = smf.ols(formula, data=df_model).fit()

    # Obtain robust covariance (HC3)
    try:
        robust_results = results.get_robustcov_results(cov_type='HC3')
    except Exception:
        # Fallback: if getting robust results fails, return the original results
        robust_results = results

    # Print a compact summary to assist users; also return the full results object
    try:
        print(robust_results.summary())
    except Exception:
        pass

    return robust_results