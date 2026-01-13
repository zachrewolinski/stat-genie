from typing import Any, Dict, List, Optional
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Helper: search dataframe columns for likely matches to a list of aliases
def _find_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    # Exact match (case-insensitive)
    for a in aliases:
        if a is None:
            continue
        a_low = a.lower()
        if a_low in cols_lower:
            return cols_lower[a_low]
    # Substring match (alias contained in column name)
    for a in aliases:
        if a is None:
            continue
        a_low = a.lower()
        for col_low, col_orig in cols_lower.items():
            if a_low in col_low:
                return col_orig
    # Pattern matches for featureN variations (featureN, feature_N, featN, feat_N)
    for a in aliases:
        if a is None:
            continue
        # if alias like 'feature14' try variations
        m = re.match(r'^(?:feature|feat|f)?_?(\d{1,3})$', a.lower())
        if m:
            n = m.group(1)
            patterns = [f'feature{n}', f'feature_{n}', f'feat{n}', f'feat_{n}', f'f{n}']
            for p in patterns:
                if p in cols_lower:
                    return cols_lower[p]
            # also try substring
            for col_low, col_orig in cols_lower.items():
                for p in patterns:
                    if p in col_low:
                        return col_orig
    return None

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataframe into final analysis-ready dataframe.

    Produces required final columns (names must match exactly):
      - AvgScore
      - STR_log
      - ExpenditurePerStudent
      - ComputersPerClassroom
      - PercentCalWorks
      - PercentReducedLunch
      - PercentEnglishLearners
      - AvgIncome
      - GradeSpan_KK08
      - County (plus County_* dummies)

    The function is robust to different source column namings by searching
    for plausible aliases for the original 'featureX' columns.
    """
    df = df.copy()

    # Define aliases for the source features we expect (allow multiple possible names)
    aliases: Dict[str, List[str]] = {
        # Dependent variable: reading (feature14) and math (feature15)
        'feature14': ['feature14', 'feature_14', 'feat14', 'feat_14', 'f14', 'reading', 'reading_score', 'read', 'read_score', 'stanford_reading', 'r15'],
        'feature15': ['feature15', 'feature_15', 'feat15', 'feat_15', 'f15', 'math', 'math_score', 'mat', 'maths', 'stanford_math'],

        # Enrollment and teachers (for student-teacher ratio)
        'feature6': ['feature6', 'feature_6', 'feat6', 'feat_6', 'f6', 'enrollment', 'enrolled', 'students', 'num_students'],
        'feature7': ['feature7', 'feature_7', 'feat7', 'feat_7', 'f7', 'num_teachers', 'teachers', 'staff_teachers'],

        # Controls
        'feature11': ['feature11', 'feature_11', 'feat11', 'feat_11', 'f11', 'expenditure', 'expenditure_per_student', 'exp_per_student', 'expenditureperstudent'],
        # include 'computer' and variants
        'feature10': ['feature10', 'feature_10', 'feat10', 'feat_10', 'f10', 'computers', 'computer', 'computers_per_classroom', 'computersperclassroom', 'computer_per_classroom', 'num_computers'],
        'feature8':  ['feature8', 'feature_8', 'feat8', 'feat_8', 'f8', 'calworks', 'percent_calworks', 'pct_calworks'],
        # include 'lunch' and variants
        'feature9':  ['feature9', 'feature_9', 'feat9', 'feat_9', 'f9', 'reduced_lunch', 'percent_reduced_lunch', 'pct_reduced_lunch', 'reduced', 'lunch'],
        # include 'english' and variants
        'feature13': ['feature13', 'feature_13', 'feat13', 'feat_13', 'f13', 'english_learners', 'percent_english_learners', 'ell', 'pct_english_learners', 'english'],
        'feature12': ['feature12', 'feature_12', 'feat12', 'feat_12', 'f12', 'avg_income', 'average_income', 'income'],
        # include 'grades' and variants for grade span
        'feature5':  ['feature5', 'feature_5', 'feat5', 'feat_5', 'f5', 'grade_span', 'gradespan', 'grades', 'grade'],
        'feature4':  ['feature4', 'feature_4', 'feat4', 'feat_4', 'f4', 'county']
    }

    # Find actual column names in df for each required source feature
    found: Dict[str, Optional[str]] = {}
    for key, als in aliases.items():
        found[key] = _find_column(df, als)

    # Helper to raise consistent error if critical source columns are missing
    def _require(col_keys: List[str]):
        missing = [k for k in col_keys if found.get(k) is None]
        if missing:
            raise KeyError(
                "Could not find required source columns for: "
                + ", ".join(missing)
                + ". Available columns: " + ", ".join(df.columns.tolist())
            )

    # Require enrollment and teachers and score columns and core controls exist
    _require(['feature6', 'feature7', 'feature14', 'feature15',
              'feature11', 'feature10', 'feature8', 'feature9',
              'feature13', 'feature12', 'feature5', 'feature4'])

    # Coerce relevant source columns to numeric where appropriate
    # Reading and math
    read_col = found['feature14']
    math_col = found['feature15']
    df[read_col] = pd.to_numeric(df[read_col], errors='coerce')
    df[math_col] = pd.to_numeric(df[math_col], errors='coerce')

    # Create dependent variable AvgScore: mean of reading and math
    df['AvgScore'] = df[[read_col, math_col]].mean(axis=1)

    # Enrollment and teachers -> StudentTeacherRatio
    enroll_col = found['feature6']
    teach_col = found['feature7']
    df[enroll_col] = pd.to_numeric(df[enroll_col], errors='coerce')
    df[teach_col] = pd.to_numeric(df[teach_col], errors='coerce')
    df['Enrollment'] = df[enroll_col]
    df['NumTeachers'] = df[teach_col]
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']
    df.loc[~np.isfinite(df['StudentTeacherRatio']), 'StudentTeacherRatio'] = np.nan
    # Remove non-positive ratios (cannot log)
    df.loc[df['StudentTeacherRatio'] <= 0, 'StudentTeacherRatio'] = np.nan
    df['STR_log'] = np.log(df['StudentTeacherRatio'])

    # Controls: coerce and rename to required final column names
    def _coerce_and_assign(src_key: str, final_name: str):
        src = found[src_key]
        if src is None:
            # Shouldn't happen due to earlier _require, but guard defensively
            df[final_name] = np.nan
        else:
            df[src] = pd.to_numeric(df[src], errors='coerce')
            df[final_name] = df[src]

    _coerce_and_assign('feature11', 'ExpenditurePerStudent')
    _coerce_and_assign('feature10', 'ComputersPerClassroom')
    _coerce_and_assign('feature8', 'PercentCalWorks')
    _coerce_and_assign('feature9', 'PercentReducedLunch')
    _coerce_and_assign('feature13', 'PercentEnglishLearners')
    _coerce_and_assign('feature12', 'AvgIncome')

    # Grade span indicator (feature5). Map to 1 for KK-08 (or K-8 variants), 0 otherwise.
    grade_span_src = found['feature5']

    def _map_gradespan(val: Any) -> int:
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        # Normalize common separators
        s_clean = re.sub(r'[\s\._]+', '-', s)
        # Look for patterns indicating K or KK and 8 (e.g., 'k-8', 'kk-08', 'k8', 'kk8', 'kk08', 'k-08')
        if re.search(r'\b(k{1,2})[^0-9a-zA-Z]*0?8\b', s_clean):
            return 1
        if re.search(r'\b0?8\b', s_clean) and ('k' in s_clean or 'kk' in s_clean):
            return 1
        # also accept explicit 'k-8' or 'k8' spelled variations
        if 'k-8' in s_clean or 'k8' in s_clean or 'kk-8' in s_clean or 'kk8' in s_clean:
            return 1
        # treat explicit 'kk-06' or containing '06' as 0; default 0 otherwise
        return 0

    if grade_span_src is not None:
        df['GradeSpan_KK08'] = df[grade_span_src].apply(_map_gradespan)
    else:
        df['GradeSpan_KK08'] = 0

    # County original column and dummies
    county_src = found['feature4']
    if county_src is not None:
        df['County'] = df[county_src].astype(str)
    else:
        df['County'] = 'Unknown'
    county_dummies = pd.get_dummies(df['County'], prefix='County', dummy_na=False, drop_first=True)
    df = pd.concat([df, county_dummies], axis=1)

    # Final required columns list (must be present)
    required_final = [
        'AvgScore',
        'STR_log',
        'ExpenditurePerStudent',
        'ComputersPerClassroom',
        'PercentCalWorks',
        'PercentReducedLunch',
        'PercentEnglishLearners',
        'AvgIncome',
        'GradeSpan_KK08'
    ]
    # Drop rows missing any of these required variables
    df = df.dropna(subset=required_final)

    # Return final dataframe (may contain helper columns; that is allowed)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit OLS regression of AvgScore on STR_log and controls including county fixed effects.

    Returns the fitted statsmodels regression results object with HC3 robust SEs.
    """
    # Identify county dummy columns created in transform (prefix 'County_')
    county_cols = [col for col in df.columns if col.startswith('County_')]

    X_cols = [
        'STR_log',
        'ExpenditurePerStudent',
        'ComputersPerClassroom',
        'PercentCalWorks',
        'PercentReducedLunch',
        'PercentEnglishLearners',
        'AvgIncome',
        'GradeSpan_KK08'
    ] + county_cols

    # Ensure all X columns exist in df
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError("Missing columns required for model: " + ", ".join(missing))

    X = df[X_cols].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['AvgScore'].astype(float)

    model_res = sm.OLS(y, X).fit(cov_type='HC3')
    return model_res