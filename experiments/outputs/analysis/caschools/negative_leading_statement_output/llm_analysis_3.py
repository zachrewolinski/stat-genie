from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/negative_leading_statement_output/caschools.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Core required columns for DV and IV: read, math, students, teachers, county
    # Drop rows missing these core values (grades not required to preserve sample; Grades_KK08 will be derived)
    df = df.dropna(subset=['read', 'math', 'students', 'teachers', 'county'])

    # Remove impossible teacher counts (avoid division by zero)
    df = df[df['teachers'].astype(float) > 0]

    # Dependent variable: average of read and math
    df['AvgScore'] = (df['read'].astype(float) + df['math'].astype(float)) / 2.0

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['students'].astype(float) / df['teachers'].astype(float)

    # Winsorize the ratio at 1st and 99th percentiles to reduce influence of extreme outliers
    if len(df) >= 3:
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower, upper)

    # Add a quadratic term to allow nonlinearity
    df['StudentTeacherRatio_sq'] = df['StudentTeacherRatio'] ** 2

    # Control: log of total students (size). Replace non-positive students with NaN to avoid -inf
    students_float = df['students'].astype(float)
    students_float = students_float.where(students_float > 0, np.nan)
    df['LogStudents'] = np.log(students_float)

    # Control: grade-span indicator (1 if KK-08, 0 otherwise). If 'grades' missing, default to 0.
    if 'grades' in df.columns:
        df['Grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)
    else:
        df['Grades_KK08'] = 0

    # For other control variables, ensure columns exist and impute missing values with the column median to preserve sample size
    for col in ['expenditure', 'income', 'calworks', 'lunch', 'english', 'computer']:
        if col not in df.columns:
            df[col] = np.nan
        # ensure numeric dtype
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median = df[col].median()
        # If median is NaN (e.g., column entirely missing), fillna will leave NaNs; that's acceptable
        df[col] = df[col].fillna(median)

    # Ensure 'county' column is present (should be from the initial dropna). If not, create placeholder (will be caught later)
    if 'county' not in df.columns:
        df['county'] = np.nan

    # Reset index for convenience
    df = df.reset_index(drop=True)

    # Ensure final dataframe contains the exact required columns (the modeling function will check existence)
    required_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'StudentTeacherRatio_sq',
        'LogStudents',
        'expenditure',
        'income',
        'calworks',
        'lunch',
        'english',
        'computer',
        'Grades_KK08',
        'county'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fits an OLS model testing whether lower student-teacher ratio (StudentTeacherRatio)
    is associated with higher average scores (AvgScore), controlling for district
    socioeconomic and resource covariates and county fixed effects.

    Returns a robust-results object and key coefficient + p-value for StudentTeacherRatio.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    # Ensure the expected columns exist
    required = ['AvgScore', 'StudentTeacherRatio', 'StudentTeacherRatio_sq', 'LogStudents',
                'expenditure', 'income', 'calworks', 'lunch', 'english', 'computer', 'Grades_KK08', 'county']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Specify model formula. County entered as categorical fixed effect.
    formula = (
        'AvgScore ~ StudentTeacherRatio + StudentTeacherRatio_sq + LogStudents + '
        'expenditure + income + calworks + lunch + english + computer + Grades_KK08 + C(county)'
    )

    # Fit OLS
    mod = smf.ols(formula=formula, data=df)
    fit = mod.fit()

    # Obtain robust (HC3) standard errors for inference
    results_robust = fit.get_robustcov_results(cov_type='HC3')

    # Print summary to console (helpful during interactive runs)
    try:
        print(results_robust.summary())
    except Exception:
        # If printing the summary fails for any reason, continue silently
        pass

    # Helper to convert params/pvalues to a pandas Series with names as index
    def _series_from_results_attr(results, attr_name):
        attr = getattr(results, attr_name, None)
        if attr is None:
            return pd.Series(dtype=float)
        # If it's already a pandas Series, ensure it has a proper index
        if isinstance(attr, pd.Series):
            return attr
        # If it's a numpy array, use model exog names as index if available
        if isinstance(attr, np.ndarray):
            try:
                names = results.model.exog_names
            except Exception:
                names = [f'var{i}' for i in range(len(attr))]
            return pd.Series(attr, index=names)
        # Fallback: try to coerce to Series
        try:
            return pd.Series(attr)
        except Exception:
            return pd.Series(dtype=float)

    params = _series_from_results_attr(results_robust, 'params')
    pvalues = _series_from_results_attr(results_robust, 'pvalues')

    # Extract coefficient and p-value for the primary independent variable
    coef = params.get('StudentTeacherRatio', np.nan)
    pval = pvalues.get('StudentTeacherRatio', np.nan)

    # Return the robust results object along with the key coefficient and p-value
    return {
        'results': results_robust,
        'coef_student_teacher_ratio': float(coef) if not pd.isna(coef) else None,
        'pval_student_teacher_ratio': float(pval) if not pd.isna(pval) else None
    }