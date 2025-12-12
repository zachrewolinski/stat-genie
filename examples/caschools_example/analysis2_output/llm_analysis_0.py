from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


def _find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first column name from candidates that exists in df (case-insensitive).
    If none found, return None.
    """
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling. The function will:
    - Locate raw columns that correspond to the conceptual variables (robust to common name variants)
    - Ensure numeric columns are numeric
    - Drop rows missing core variables (enrollment, teachers, reading, math)
    - Remove rows with non-positive teacher counts
    - Create StudentTeacherRatio = Enrollment / NumTeachers
    - Create AvgScore = mean(reading, math)
    - Rename / copy relevant control variables to deterministic column names
    - Return only the columns used in the model (in a stable order)
    """
    df = df.copy()

    # Candidate names for the core raw columns (robust matching)
    enroll_cands = [
        'feature6', 'enrollment', 'enroll', 'total_enrollment', 'totalenrollment', 'ENROLL', 'Enrollment',
        'students', 'student', 'students_total', 'enrolled'
    ]
    teachers_cands = [
        'feature7', 'numteachers', 'num_teachers', 'teachers', 'Teachers', 'NumTeachers', 'teacher_count', 'teacher'
    ]
    reading_cands = [
        'feature14', 'reading', 'read', 'avg_reading', 'reading_avg', 'AvgReading', 'AverageReading'
    ]
    math_cands = [
        'feature15', 'math', 'Math', 'avg_math', 'math_avg', 'AvgMath', 'AverageMath'
    ]

    # Find core columns
    enroll_col = _find_first_column(df, enroll_cands)
    teachers_col = _find_first_column(df, teachers_cands)
    reading_col = _find_first_column(df, reading_cands)
    math_col = _find_first_column(df, math_cands)

    # If any core columns are missing, raise a clear error
    if not all([enroll_col, teachers_col, reading_col, math_col]):
        available = list(df.columns)
        missing = []
        if not enroll_col:
            missing.append('enrollment (candidates: {})'.format(enroll_cands))
        if not teachers_col:
            missing.append('num teachers (candidates: {})'.format(teachers_cands))
        if not reading_col:
            missing.append('reading score (candidates: {})'.format(reading_cands))
        if not math_col:
            missing.append('math score (candidates: {})'.format(math_cands))
        raise ValueError(f"Could not find required raw columns in the input dataframe. Missing: {missing}. Available columns: {available}")

    # Convert core columns to numeric (coerce errors to NaN)
    df[enroll_col] = pd.to_numeric(df[enroll_col], errors='coerce')
    df[teachers_col] = pd.to_numeric(df[teachers_col], errors='coerce')
    df[reading_col] = pd.to_numeric(df[reading_col], errors='coerce')
    df[math_col] = pd.to_numeric(df[math_col], errors='coerce')

    # Drop rows missing the core variables needed to compute the IV and DV
    df = df.dropna(subset=[enroll_col, teachers_col, reading_col, math_col])

    # Remove rows with non-positive number of teachers to avoid division errors
    df = df[df[teachers_col] > 0]

    # Create clear column names for modeling (these exact names are required by contract)
    df['Enrollment'] = df[enroll_col]
    df['NumTeachers'] = df[teachers_col]
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[[reading_col, math_col]].mean(axis=1)

    # Controls mapping: try to find plausible source columns for each control; if not found, create NA column
    control_candidates = {
        'PercentCalWorks': ['feature8', 'calworks', 'percent_calworks', 'percentcalworks', 'CalWorks', 'PercentCalWorks'],
        'PercentReducedLunch': ['feature9', 'reduced_lunch', 'percent_reduced_lunch', 'PercentReducedLunch', 'lunch', 'free_reduced_lunch'],
        'NumComputers': ['feature10', 'numcomputers', 'num_computers', 'computers', 'NumComputers', 'computer', 'computers_count'],
        'ExpenditurePerStudent': ['feature11', 'expenditureperstudent', 'expenditure_per_student', 'expend_per_student', 'ExpenditurePerStudent', 'expenditure'],
        'AvgIncome': ['feature12', 'avgincome', 'avg_income', 'average_income', 'AvgIncome', 'income'],
        'PercentEnglishLearners': ['feature13', 'english_learners', 'percent_english_learners', 'PercentEnglishLearners', 'english']
    }

    for out_col, cands in control_candidates.items():
        src = _find_first_column(df, cands)
        if src is not None:
            # convert to numeric
            df[out_col] = pd.to_numeric(df[src], errors='coerce')
        else:
            df[out_col] = np.nan

    # Categorical controls: county and grade span
    county_cands = ['feature4', 'county', 'County', 'COUNTY']
    gradespan_cands = ['feature5', 'gradespan', 'GradeSpan', 'grade_span', 'GradeSpanType', 'grades', 'gradespan_type', 'gradespan']

    county_src = _find_first_column(df, county_cands)
    gradespan_src = _find_first_column(df, gradespan_cands)

    if county_src is not None:
        df['County'] = df[county_src].astype(str)
    else:
        df['County'] = ''  # empty string indicates missing category

    if gradespan_src is not None:
        df['GradeSpan'] = df[gradespan_src].astype(str)
    else:
        df['GradeSpan'] = ''

    # Keep only rows with finite StudentTeacherRatio and AvgScore
    df = df[df['StudentTeacherRatio'].replace([np.inf, -np.inf], np.nan).notna()]
    df = df[df['AvgScore'].replace([np.inf, -np.inf], np.nan).notna()]

    # Return just the columns needed for modeling (in stable order)
    out_cols = [
        'StudentTeacherRatio', 'AvgScore',
        'PercentCalWorks', 'PercentReducedLunch', 'NumComputers', 'ExpenditurePerStudent',
        'AvgIncome', 'PercentEnglishLearners', 'Enrollment', 'NumTeachers',
        'County', 'GradeSpan'
    ]
    # Ensure all out_cols exist in df (they should, but create missing as NA to be safe)
    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[out_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for district characteristics.
    Uses robust (heteroskedasticity-consistent) standard errors (HC3).

    Model formula:
      AvgScore ~ StudentTeacherRatio + PercentCalWorks + PercentReducedLunch + NumComputers
                 + ExpenditurePerStudent + AvgIncome + PercentEnglishLearners + Enrollment
                 + C(County) + C(GradeSpan)

    Returns the fitted statsmodels RegressionResults object with robust covariance (HC3).
    """
    # Ensure the expected columns are present
    required = ['AvgScore', 'StudentTeacherRatio', 'PercentCalWorks', 'PercentReducedLunch',
                'NumComputers', 'ExpenditurePerStudent', 'AvgIncome', 'PercentEnglishLearners',
                'Enrollment', 'County', 'GradeSpan']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    formula = (
        'AvgScore ~ StudentTeacherRatio + PercentCalWorks + PercentReducedLunch + NumComputers '
        '+ ExpenditurePerStudent + AvgIncome + PercentEnglishLearners + Enrollment '
        '+ C(County) + C(GradeSpan)'
    )

    # Fit OLS and then compute robust (HC3) covariance results
    ols_mod = smf.ols(formula=formula, data=df)
    res = ols_mod.fit()
    robust_res = res.get_robustcov_results(cov_type='HC3')

    # Print a brief summary and return the fitted model with robust covariance
    print(robust_res.summary())
    return robust_res