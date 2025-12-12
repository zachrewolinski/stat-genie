from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first column name from df.columns that matches any candidate (case-insensitive).
    Performs exact (case-insensitive) match first then substring match.
    """
    if not candidates:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        cand_l = str(cand).lower().strip()
        if cand_l in cols_lower:
            return cols_lower[cand_l]
    # Fallback: substring match (prefer exact token matches)
    for cand in candidates:
        cand_l = str(cand).lower().strip()
        for col in df.columns:
            if cand_l in col.lower():
                return col
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # If the final columns are already present, assume data is already transformed.
    final_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ExpenditurePerStudent', 'PercentCalWorks',
        'PercentReducedPriceLunch', 'PercentEnglishLearners', 'DistrictAvgIncome',
        'GradeSpan_KK08', 'LogTotalEnrollment'
    ]
    if all(col in df.columns for col in final_cols):
        # Ensure numeric types for model columns
        for col in final_cols:
            if col != 'GradeSpan_KK08':  # GradeSpan_KK08 can be int/bool
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # Define candidate raw column names for each required conceptual variable.
    candidates: Dict[str, List[str]] = {
        'TotalEnrollment': ['feature6', 'enroll', 'enrollment', 'total_enrollment', 'totalenrollment', 'ENROLL', 'total enrollment'],
        'NumTeachers': ['feature7', 'teachers', 'num_teachers', 'numteachers', 'TEACHERS', 'num teacher'],
        'NumComputers': ['feature10', 'computers', 'num_computers', 'numcomputers', 'comput'],
        'ExpenditurePerStudent': ['feature11', 'expenditure_per_student', 'expenditure', 'exp_per_student', 'exppp', 'expenditure per student'],
        'DistrictAvgIncome': ['feature12', 'avg_income', 'district_avg_income', 'avginc', 'AvgIncome', 'average income'],
        'PercentCalWorks': ['feature8', 'calworks', 'percent_calworks', 'PercentCalWorks', 'cal works'],
        'PercentReducedPriceLunch': ['feature9', 'reduced_price_lunch', 'percent_reduced_price_lunch', 'reducedlunch', 'PercentReducedPriceLunch', 'reduced price lunch'],
        'PercentEnglishLearners': ['feature13', 'ell', 'english_learners', 'percent_english_learners', 'PercentEnglishLearners'],
        'GradeSpan': ['feature5', 'grade_span', 'gradespan', 'GradeSpan', 'grade span'],
        'ReadingScore': ['feature14', 'read', 'reading', 'read_score', 'average_reading', 'avg_read'],
        'MathScore': ['feature15', 'math', 'math_score', 'average_math', 'avg_math'],
        'AvgScore': ['avgscore', 'avg_score', 'avg', 'average_score', 'AverageScore']
    }

    # Locate columns
    located: Dict[str, Optional[str]] = {}
    for key, cands in candidates.items():
        located[key] = _find_column(df, cands)

    # If AvgScore exists directly in raw df, use it.
    if located.get('AvgScore') is not None:
        df['AvgScore'] = pd.to_numeric(df[located['AvgScore']], errors='coerce')
    else:
        # Need reading and math scores to compute AvgScore
        read_col = located.get('ReadingScore')
        math_col = located.get('MathScore')

        # If either reading or math is missing, try to find any two plausible numeric score columns
        if read_col is None or math_col is None:
            # Look for columns likely corresponding to reading/math by scanning column names
            possible_scores = []
            for col in df.columns:
                lname = col.lower()
                if any(tok in lname for tok in ['read', 'math', 'score', 'avg', 'test']):
                    possible_scores.append(col)
            # Keep only numeric-convertible ones
            numeric_scores = []
            for col in possible_scores:
                ser = pd.to_numeric(df[col], errors='coerce')
                if ser.notna().sum() > 0:
                    numeric_scores.append(col)
            # Choose two distinct columns
            numeric_scores = list(dict.fromkeys(numeric_scores))  # preserve order, unique
            if len(numeric_scores) >= 2:
                read_col, math_col = numeric_scores[0], numeric_scores[1]
            else:
                # As a last resort, try to use any two numeric columns excluding clearly non-score fields
                numeric_candidates = []
                for col in df.columns:
                    ser = pd.to_numeric(df[col], errors='coerce')
                    if ser.notna().sum() > 0:
                        numeric_candidates.append(col)
                # Remove columns we will use for controls (to avoid picking them as scores)
                exclude_keywords = ['enroll', 'teacher', 'comput', 'exp', 'income', 'calwork', 'ell', 'reduce', 'grade', 'log']
                numeric_candidates = [c for c in numeric_candidates if not any(k in c.lower() for k in exclude_keywords)]
                numeric_candidates = list(dict.fromkeys(numeric_candidates))
                if len(numeric_candidates) >= 2:
                    read_col, math_col = numeric_candidates[0], numeric_candidates[1]
                else:
                    raise KeyError(
                        "Could not locate reading and math score columns (feature14 & feature15). "
                        "Ensure the input DataFrame contains score columns or AvgScore."
                    )

        # Convert and compute AvgScore
        df[read_col] = pd.to_numeric(df[read_col], errors='coerce')
        df[math_col] = pd.to_numeric(df[math_col], errors='coerce')
        df['AvgScore'] = df[[read_col, math_col]].mean(axis=1)

    # Locate and convert other raw columns; raise error if essential ones missing
    raw_mappings = {
        'TotalEnrollment': ['feature6', 'TotalEnrollment'],
        'NumTeachers': ['feature7', 'NumTeachers'],
        'NumComputers': ['feature10', 'NumComputers'],
        'ExpenditurePerStudent': ['feature11', 'ExpenditurePerStudent'],
        'DistrictAvgIncome': ['feature12', 'DistrictAvgIncome'],
        'PercentCalWorks': ['feature8', 'PercentCalWorks'],
        'PercentReducedPriceLunch': ['feature9', 'PercentReducedPriceLunch'],
        'PercentEnglishLearners': ['feature13', 'PercentEnglishLearners'],
    }

    # For each conceptual variable, attempt to map to an existing raw column (using candidates above)
    mapped_cols: Dict[str, str] = {}
    for concept, cands in raw_mappings.items():
        # Prefer previously located column if found
        raw_col = located.get(concept)
        if raw_col is None:
            # Try the generic candidate list first, then the raw_mappings list
            raw_col = _find_column(df, candidates.get(concept, []) or cands)
            if raw_col is None:
                raw_col = _find_column(df, cands)

        # If still not found, try a heuristic fallback:
        if raw_col is None:
            # Heuristic: choose a numeric column that looks like counts (large median)
            possible = []
            for col in df.columns:
                # Skip already-constructed or obvious non-raw columns
                if col in ['AvgScore', 'StudentTeacherRatio', 'LogTotalEnrollment', 'GradeSpan_KK08', 'ComputersPerStudent']:
                    continue
                ser = pd.to_numeric(df[col], errors='coerce')
                if ser.notna().sum() < max(1, int(0.1 * len(df))):
                    continue
                lower = col.lower()
                # For enrollment, avoid percent/score/income-like names
                if any(k in lower for k in ['percent', 'pct', '%', 'score', 'avg', 'income', 'math', 'read', 'grade', 'ell', 'calwork', 'lunch']):
                    continue
                median_val = float(ser.median(skipna=True)) if ser.notna().sum() > 0 else 0.0
                possible.append((col, median_val))
            if possible:
                # pick the column with the largest median (likely enrollment or expenditure depending on concept)
                possible.sort(key=lambda x: x[1], reverse=True)
                raw_col = possible[0][0]

        if raw_col is None:
            raise KeyError(f"Could not locate a column for required variable '{concept}'. Searched candidates.")

        # Convert to numeric where appropriate
        df[raw_col] = pd.to_numeric(df[raw_col], errors='coerce')
        mapped_cols[concept] = raw_col

    # Assign final column names according to the contract
    df['TotalEnrollment'] = df[mapped_cols['TotalEnrollment']]
    df['NumTeachers'] = df[mapped_cols['NumTeachers']]
    df['NumComputers'] = df[mapped_cols['NumComputers']]
    df['ExpenditurePerStudent'] = df[mapped_cols['ExpenditurePerStudent']]
    df['DistrictAvgIncome'] = df[mapped_cols['DistrictAvgIncome']]
    df['PercentCalWorks'] = df[mapped_cols['PercentCalWorks']]
    df['PercentReducedPriceLunch'] = df[mapped_cols['PercentReducedPriceLunch']]
    df['PercentEnglishLearners'] = df[mapped_cols['PercentEnglishLearners']]

    # Student-teacher ratio (students per teacher). Avoid division by zero/invalid teachers.
    df.loc[(df['NumTeachers'] <= 0) | (df['NumTeachers'].isna()), 'NumTeachers'] = np.nan
    df['StudentTeacherRatio'] = df['TotalEnrollment'] / df['NumTeachers']

    # Log-transformed enrollment (natural log). Replace nonpositive totals with NaN.
    df.loc[(df['TotalEnrollment'] <= 0) | (df['TotalEnrollment'].isna()), 'TotalEnrollment'] = np.nan
    df['LogTotalEnrollment'] = np.log(df['TotalEnrollment'])

    # Grade span indicator: 1 if KK-08, 0 if KK-06 or otherwise 0.
    grade_col = located.get('GradeSpan') or _find_column(df, ['feature5', 'GradeSpan', 'grade_span', 'gradespan', 'grade span'])
    if grade_col is not None:
        # Work on a string-safe version of the grade column
        df['GradeSpan_KK08'] = df[grade_col].astype(str).str.strip().apply(lambda x: 1 if x == 'KK-08' else 0)
    else:
        # If we cannot find a grade span column, default to 0 (KK-06 or unknown)
        df['GradeSpan_KK08'] = 0

    # Optionally compute computers per student as a descriptive variable
    df['ComputersPerStudent'] = df['NumComputers'] / df['TotalEnrollment']

    # Keep only rows with the necessary data for the model
    required_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ExpenditurePerStudent', 'PercentCalWorks',
        'PercentReducedPriceLunch', 'PercentEnglishLearners', 'DistrictAvgIncome',
        'GradeSpan_KK08', 'LogTotalEnrollment'
    ]

    df = df.dropna(subset=required_cols)

    # Return the dataframe with the new columns
    return df


def model(df: pd.DataFrame) -> Any:
    # Build design matrix X and outcome y using the transformed dataframe columns
    X_cols = [
        'StudentTeacherRatio',
        'ExpenditurePerStudent',
        'PercentCalWorks',
        'PercentReducedPriceLunch',
        'PercentEnglishLearners',
        'DistrictAvgIncome',
        'GradeSpan_KK08',
        'LogTotalEnrollment'
    ]

    # Verify required columns exist
    missing = [c for c in X_cols + ['AvgScore'] if c not in df.columns]
    if missing:
        raise KeyError(f"The input dataframe is missing required columns for the model: {missing}")

    X = df[X_cols].copy()
    # Ensure numeric types
    X = X.apply(pd.to_numeric, errors='coerce')
    X = sm.add_constant(X)
    y = pd.to_numeric(df['AvgScore'], errors='coerce')

    # Fit OLS with robust (HC3) standard errors to be resilient to heteroskedasticity
    model_res = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

    # Print a short summary and return the fitted model results object
    try:
        print(model_res.summary())
    except Exception:
        pass

    return model_res