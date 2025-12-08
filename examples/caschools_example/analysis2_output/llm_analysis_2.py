from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find the first matching column name in df from a list of candidate names.
    Matches are case-insensitive.
    Returns the actual column name from df if found, otherwise None.
    """
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        # Direct exact match first
        if cand in df.columns:
            return cand
        # Case-insensitive match
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.
    Produces the following final columns (these exact names are required downstream):
      - StudentTeacherRatio_clipped: Enrollment / NumTeachers, clipped at 1st and 99th percentiles
      - AvgScore: mean of ReadingScore and MathScore (computed from whatever scores are available)
      - NumComputers, ExpenditurePerStudent, DistrictIncomeK, PercEnglishLearners,
        PercReducedLunch, PercCalWorks, County, GradeSpan, Enrollment, NumTeachers

    The function is robust to a variety of raw column namings by searching for common aliases.
    """
    df = df.copy()

    # Define candidate raw names for each conceptual final column (common variants)
    alias_map: Dict[str, List[str]] = {
        'Enrollment': ['feature6', 'Feature6', 'enrollment', 'enroll', 'students', 'total_enrollment'],
        'NumTeachers': ['feature7', 'Feature7', 'numteachers', 'teachers', 'num_teachers'],
        'NumComputers': ['feature10', 'Feature10', 'numcomputers', 'computers'],
        'ExpenditurePerStudent': ['feature11', 'Feature11', 'expenditureperstudent', 'expenditure_per_student'],
        'DistrictIncomeK': ['feature12', 'Feature12', 'districtincomek', 'district_income_k', 'incomek'],
        'PercEnglishLearners': ['feature13', 'Feature13', 'percenglishlearners', 'perc_english_learners', 'englishlearners'],
        'PercReducedLunch': ['feature9', 'Feature9', 'percreducedlunch', 'perc_reduced_lunch', 'reduced_lunch'],
        'PercCalWorks': ['feature8', 'Feature8', 'perccalworks', 'perc_cal_works', 'calworks'],
        'ReadingScore': ['feature14', 'Feature14', 'readingscore', 'reading_score', 'reading'],
        'MathScore': ['feature15', 'Feature15', 'mathscore', 'math_score', 'math'],
        'County': ['feature4', 'Feature4', 'county'],
        'GradeSpan': ['feature5', 'Feature5', 'gradespan', 'grade_span', 'grade']
    }

    # Build a rename map from actual raw column names to the required final column names
    rename_map: Dict[str, str] = {}
    for final_name, candidates in alias_map.items():
        found = _find_column(df, candidates)
        if found is not None:
            rename_map[found] = final_name

    # Apply renaming for any found columns
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure numeric columns are numeric (coerce non-numeric to NaN)
    numeric_cols = [
        'Enrollment', 'NumTeachers', 'NumComputers', 'ExpenditurePerStudent',
        'DistrictIncomeK', 'PercEnglishLearners', 'PercReducedLunch',
        'PercCalWorks', 'ReadingScore', 'MathScore'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # At this point, Enrollment and NumTeachers should exist (renamed); if not, create as NaN
    # so that final dataframe always contains the required columns.
    if 'Enrollment' not in df.columns:
        df['Enrollment'] = np.nan
    if 'NumTeachers' not in df.columns:
        df['NumTeachers'] = np.nan

    # Compute student-teacher ratio; guard against division by zero and missing values
    # Create an intermediate column StudentTeacherRatio (not part of final required names)
    df['StudentTeacherRatio'] = np.where(
        (df['NumTeachers'].notna()) & (df['NumTeachers'] != 0),
        df['Enrollment'] / df['NumTeachers'],
        np.nan
    )

    # Compute average score from reading and math (AvgScore is required final name)
    # Use available scores: take mean across the two, skipping NaNs so that if one score is present we use it.
    if ('ReadingScore' in df.columns) or ('MathScore' in df.columns):
        # Ensure missing score columns are treated as NaN if they don't exist
        cols_for_avg = [c for c in ['ReadingScore', 'MathScore'] if c in df.columns]
        if cols_for_avg:
            df['AvgScore'] = df[cols_for_avg].mean(axis=1, skipna=True)
        else:
            df['AvgScore'] = np.nan
    else:
        df['AvgScore'] = np.nan

    # Replace infinite values and drop rows missing the core variables needed to compute ratio and outcome
    df = df.replace([np.inf, -np.inf], np.nan)
    # Keep only rows where both StudentTeacherRatio and AvgScore are present (model needs both)
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgScore'])

    # Clip extreme StudentTeacherRatio values at 1st and 99th percentiles to reduce influence of outliers
    if not df.empty:
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        if pd.isna(lower) or pd.isna(upper):
            df['StudentTeacherRatio_clipped'] = df['StudentTeacherRatio']
        else:
            df['StudentTeacherRatio_clipped'] = df['StudentTeacherRatio'].clip(lower, upper)
    else:
        # No rows to operate on; create the column to preserve schema
        df['StudentTeacherRatio_clipped'] = pd.Series(dtype=float)

    # Fill missing control variables with median (simple imputation)
    control_cols = [
        'NumComputers', 'ExpenditurePerStudent', 'DistrictIncomeK',
        'PercEnglishLearners', 'PercReducedLunch', 'PercCalWorks'
    ]
    for c in control_cols:
        if c in df.columns:
            median_val = df[c].median()
            if not pd.isna(median_val):
                df[c] = df[c].fillna(median_val)

    # Ensure categorical controls are present (if missing, create as NA)
    if 'GradeSpan' not in df.columns:
        df['GradeSpan'] = np.nan
    if 'County' not in df.columns:
        df['County'] = np.nan

    # Ensure all required final columns exist in the returned dataframe (create with NA if absent)
    final_cols = [
        'StudentTeacherRatio_clipped', 'AvgScore',
        'NumComputers', 'ExpenditurePerStudent', 'DistrictIncomeK',
        'PercEnglishLearners', 'PercReducedLunch', 'PercCalWorks',
        'County', 'GradeSpan', 'Enrollment', 'NumTeachers'
    ]
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Return dataframe with final columns in the specified order
    return df[final_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and average academic performance.
    Model specification:
      AvgScore ~ StudentTeacherRatio_clipped + controls + categorical GradeSpan + categorical County

    Uses heteroskedasticity-robust (HC3) standard errors.
    Returns the fitted regression results object (with robust covariance applied).
    """
    # Confirm required columns exist
    required = ['AvgScore', 'StudentTeacherRatio_clipped']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe passed to model().")

    # Build formula with controls and categorical variables if present and valid (at least 2 non-missing distinct levels)
    cat_terms: List[str] = []

    def _has_enough_levels(series: pd.Series) -> bool:
        # Return True if series has at least 2 non-missing distinct levels after stripping empty strings.
        if series.name not in df.columns:
            return False
        s = series.copy()
        # Normalize object dtype strings: trim and treat empty strings as missing
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            s = s.astype(object)
            s = s.replace(r'^\s*$', np.nan, regex=True)
        non_na = s.dropna()
        return non_na.nunique() >= 2

    if 'GradeSpan' in df.columns and _has_enough_levels(df['GradeSpan']):
        cat_terms.append('C(GradeSpan)')
    if 'County' in df.columns and _has_enough_levels(df['County']):
        cat_terms.append('C(County)')

    controls: List[str] = []
    for c in ['NumComputers', 'ExpenditurePerStudent', 'DistrictIncomeK',
              'PercEnglishLearners', 'PercReducedLunch', 'PercCalWorks']:
        # Include control only if present and has at least one non-missing value (transform may have imputed medians)
        if c in df.columns and df[c].notna().any():
            controls.append(c)

    rhs_terms = ['StudentTeacherRatio_clipped'] + controls + cat_terms
    formula = 'AvgScore ~ ' + ' + '.join(rhs_terms)

    # Before fitting, ensure there is at least one row with non-missing values for all variables used in the model
    vars_for_model = ['AvgScore', 'StudentTeacherRatio_clipped'] + controls
    # include categorical columns in the check only if they are part of the formula
    if 'C(GradeSpan)' in cat_terms:
        vars_for_model.append('GradeSpan')
    if 'C(County)' in cat_terms:
        vars_for_model.append('County')
    # keep only those that actually exist in df
    vars_for_model = [v for v in vars_for_model if v in df.columns]
    df_for_fit = df[vars_for_model].dropna()
    if df_for_fit.shape[0] == 0:
        raise ValueError("No observations available after dropping rows with missing values for model variables.")

    # Fit OLS on the cleaned dataframe and then obtain robust covariance (HC3)
    ols_model = smf.ols(formula=formula, data=df_for_fit)
    results = ols_model.fit()
    robust_results = results.get_robustcov_results(cov_type='HC3')

    return robust_results