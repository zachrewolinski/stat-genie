from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find the first column in df that matches any candidate name or contains
    one of the candidate tokens (case-insensitive). Returns None if nothing found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    # Exact match first (case-sensitive)
    for cand in candidates:
        if cand in df.columns:
            return cand
    # Exact match case-insensitive
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # Token/substring matching from candidate tokens against column names
    cand_tokens = []
    for cand in candidates:
        tokens = [t for t in str(cand).strip().lower().replace('-', ' ').replace('_', ' ').split() if t]
        cand_tokens.extend(tokens)
    # search for any column containing any token
    for col in df.columns:
        col_l = col.lower()
        for tok in cand_tokens:
            if tok and tok in col_l:
                return col
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with the variables required for modeling.

    Produces the exact final column names required by the analysis:
      - StudentTeacherRatio
      - AvgTestScore
      - PercentReducedLunch
      - PercentEnglishLearners
      - ExpenditurePerStudent
      - AvgDistrictIncome
      - ComputersPerStudent
      - GradeSpan_* dummies (prefix 'GradeSpan', drop_first=True)

    The function attempts to locate expected source columns (e.g., 'feature6') but
    will also accept commonly used alternative names if present in the input dataframe.
    """
    df = df.copy()

    # Candidate names for each anonymized feature; will pick first matching column present
    candidate_map: Dict[str, List[str]] = {
        'feature6': [
            'feature6', 'enrollment', 'enrol', 'students', 'student', 'total_students',
            'n_students', 'pupils'
        ],
        'feature7': [
            'feature7', 'teacherfte', 'teacher_fte', 'teachers', 'teacher', 'num_teachers', 'n_teachers'
        ],
        'feature9': [
            'feature9', 'percent_reduced_lunch', 'reduced_price_lunch', 'pct_reduced_lunch',
            'frl', 'percent_frl', 'lunch', 'reduced_lunch', 'free_reduced_lunch'
        ],
        'feature10': [
            'feature10', 'computers', 'computer', 'num_computers', 'computers_total',
            'computers_per_student', 'computer_per_student'
        ],
        'feature11': [
            'feature11', 'expenditure_per_student', 'expenditure', 'exp_per_student',
            'spend_per_student', 'spending_per_student'
        ],
        'feature12': [
            'feature12', 'avg_district_income', 'district_income', 'avg_income', 'median_income', 'income'
        ],
        'feature13': [
            'feature13', 'percent_english_learners', 'english_learners', 'english', 'percent_el',
            'el_percent', 'el'
        ],
        'feature14': [
            'feature14', 'avg_reading', 'reading', 'read', 'avg_reading_score', 'reading_score', 'read_score'
        ],
        'feature15': [
            'feature15', 'avg_math', 'math', 'avg_math_score', 'math_score', 'maths'
        ],
        'feature5': [
            'feature5', 'grade_span', 'gradespan', 'grade_span_type', 'grade_span_description', 'grade', 'grades'
        ]
    }

    # Locate actual column names present in df for each anonymized feature
    actual_cols: Dict[str, Optional[str]] = {}
    for feat, candidates in candidate_map.items():
        actual = _find_column(df, candidates)
        actual_cols[feat] = actual

    # Ensure essential columns are present (enrollment, teacher counts, reading, math)
    essentials = ['feature6', 'feature7', 'feature14', 'feature15']
    missing_essentials = [f for f in essentials if actual_cols.get(f) is None]
    if missing_essentials:
        raise KeyError(
            f"Input dataframe is missing required columns. Could not find any of the candidate names for: {missing_essentials}. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert located numeric columns to numeric dtype (coerce errors to NaN)
    numeric_feats = [
        'feature6', 'feature7', 'feature9', 'feature10', 'feature11',
        'feature12', 'feature13', 'feature14', 'feature15'
    ]
    for feat in numeric_feats:
        col = actual_cols.get(feat)
        if col is not None and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Now drop rows missing essential variables
    df = df.dropna(subset=[actual_cols['feature6'], actual_cols['feature7'],
                           actual_cols['feature14'], actual_cols['feature15']])

    # Remove impossible teacher counts (avoid division by zero)
    df = df[pd.to_numeric(df[actual_cols['feature7']], errors='coerce') > 0]

    # Create final required columns using located source columns
    # Enrollment and TeacherFTE
    df['Enrollment'] = df[actual_cols['feature6']].astype(float)
    df['TeacherFTE'] = df[actual_cols['feature7']].astype(float)
    df['StudentTeacherRatio'] = df['Enrollment'] / df['TeacherFTE']

    # Average test scores
    df['AvgReading'] = df[actual_cols['feature14']].astype(float)
    df['AvgMath'] = df[actual_cols['feature15']].astype(float)
    df['AvgTestScore'] = df[['AvgReading', 'AvgMath']].mean(axis=1)

    # Controls
    if actual_cols.get('feature9') is not None:
        df['PercentReducedLunch'] = df[actual_cols['feature9']].astype(float)
    else:
        df['PercentReducedLunch'] = np.nan

    if actual_cols.get('feature13') is not None:
        df['PercentEnglishLearners'] = df[actual_cols['feature13']].astype(float)
    else:
        df['PercentEnglishLearners'] = np.nan

    if actual_cols.get('feature11') is not None:
        df['ExpenditurePerStudent'] = df[actual_cols['feature11']].astype(float)
    else:
        df['ExpenditurePerStudent'] = np.nan

    if actual_cols.get('feature12') is not None:
        df['AvgDistrictIncome'] = df[actual_cols['feature12']].astype(float)
    else:
        df['AvgDistrictIncome'] = np.nan

    # Computers per student
    if actual_cols.get('feature10') is not None:
        df['ComputersPerStudent'] = pd.to_numeric(df[actual_cols['feature10']], errors='coerce') / df['Enrollment']
    else:
        df['ComputersPerStudent'] = np.nan
    df['ComputersPerStudent'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Grade-span dummies from feature5 (if present)
    if actual_cols.get('feature5') is not None:
        try:
            df[actual_cols['feature5']] = df[actual_cols['feature5']].astype('category')
            dummies = pd.get_dummies(df[actual_cols['feature5']], prefix='GradeSpan', drop_first=True)
            df = pd.concat([df, dummies], axis=1)
        except Exception:
            # If any error in creating dummies, do not fail; leave grade-span dummies out
            pass

    # Ensure final required columns exist in the dataframe (create if missing with NaN)
    required_final = [
        'StudentTeacherRatio',
        'AvgTestScore',
        'PercentReducedLunch',
        'PercentEnglishLearners',
        'ExpenditurePerStudent',
        'AvgDistrictIncome',
        'ComputersPerStudent'
    ]
    for col in required_final:
        if col not in df.columns:
            df[col] = np.nan

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgTestScore on StudentTeacherRatio and controls.

    Expects the input dataframe to already contain the final required columns produced
    by `transform`. Returns a statsmodels results object with HC3 robust standard errors.
    """
    df = df.copy()

    # Identify grade-span dummy columns (any column that starts with 'GradeSpan_')
    grade_span_cols = [c for c in df.columns if c.startswith('GradeSpan_')]

    # Define model covariates (must use exact final column names)
    base_controls = [
        'PercentReducedLunch',
        'PercentEnglishLearners',
        'ExpenditurePerStudent',
        'AvgDistrictIncome',
        'ComputersPerStudent'
    ]

    model_vars = ['StudentTeacherRatio'] + base_controls + grade_span_cols

    # Drop rows with missing values in model variables or dependent variable
    required = model_vars + ['AvgTestScore']
    df_model = df.dropna(subset=required)

    # Prepare X and y
    y = df_model['AvgTestScore'].astype(float)
    X = df_model[model_vars].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS and then get robust covariance (HC3)
    ols_res = sm.OLS(y, X).fit()
    robust_res = ols_res.get_robustcov_results(cov_type='HC3')

    return robust_res