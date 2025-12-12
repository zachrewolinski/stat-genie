from typing import Any
import numpy as np
import pandas as pd


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset to the analysis-ready dataframe.

    Outputs (columns required by the model):
      - StudentTeacherRatio: Enrollment / NumTeachers (winsorized 1st-99th pct)
      - AvgScore: mean of AvgReading and AvgMath
      - PercentCalWorks, PercentReducedLunch, PercentEnglishLearners,
        ExpenditurePerStudent, DistrictIncome, ComputersPerStudent,
        Enrollment, County, GradeSpan
    """
    df = df.copy()
    orig_cols = list(df.columns)

    # Rename raw feature columns to analysis-friendly names when present
    rename_map = {
        'feature6': 'Enrollment',
        'feature7': 'NumTeachers',
        'feature8': 'PercentCalWorks',
        'feature9': 'PercentReducedLunch',
        'feature10': 'NumComputers',
        'feature11': 'ExpenditurePerStudent',
        'feature12': 'DistrictIncome',
        'feature13': 'PercentEnglishLearners',
        'feature14': 'AvgReading',
        'feature15': 'AvgMath',
        'feature4': 'County',
        'feature5': 'GradeSpan'
    }
    # Only rename columns that actually exist to avoid KeyErrors
    existing_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_rename_map:
        df = df.rename(columns=existing_rename_map)

    # Helper to find a column name in original columns by matching keywords (case-insensitive)
    def find_col_by_keywords(columns, keywords):
        lower_cols = [c.lower() for c in columns]
        for kw in keywords:
            for orig_col, lower_col in zip(columns, lower_cols):
                if kw in lower_col:
                    return orig_col
        return None

    # Map likely alternative source column names to the standardized names if they are missing
    # Define search keywords for each conceptual variable
    fallback_map = {
        'Enrollment': ['enroll', 'students', 'enrollment'],
        'NumTeachers': ['teacher', 'numteachers', 'num_teach', 'fte'],
        'NumComputers': ['computer', 'computers', 'numcomputers'],
        'PercentCalWorks': ['calwork', 'calworks', 'cal_work'],
        'PercentReducedLunch': ['reduced', 'lunch', 'reducedlunch', 'percentreduced'],
        'PercentEnglishLearners': ['english', 'ell', 'englishlearner', 'english_learn'],
        'ExpenditurePerStudent': ['expend', 'expenditure', 'spend', 'expenditureperstudent'],
        'DistrictIncome': ['income', 'districtincome'],
        'AvgReading': ['read', 'reading'],
        'AvgMath': ['math'],
        'County': ['county'],
        'GradeSpan': ['gradespan', 'grade_span', 'grade', 'span']
    }

    # For each target, if missing, try to find a suitable source in original columns and copy it
    for target, keywords in fallback_map.items():
        if target not in df.columns:
            candidate = find_col_by_keywords(orig_cols, keywords)
            if candidate is not None and candidate in df.columns:
                df[target] = df[candidate]

    # Ensure numeric columns are numeric where present
    num_cols = [
        'Enrollment', 'NumTeachers', 'PercentCalWorks', 'PercentReducedLunch',
        'NumComputers', 'ExpenditurePerStudent', 'DistrictIncome', 'PercentEnglishLearners',
        'AvgReading', 'AvgMath'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute dependent variable: average of reading and math district means
    if 'AvgReading' in df.columns and 'AvgMath' in df.columns:
        df['AvgScore'] = df[['AvgReading', 'AvgMath']].mean(axis=1)
    else:
        # Try to find reading/math-like columns if not already present (again, robust detection)
        read_col = None
        math_col = None
        if 'AvgReading' not in df.columns:
            read_col = find_col_by_keywords(orig_cols, ['read', 'reading'])
            if read_col and read_col in df.columns:
                df['AvgReading'] = pd.to_numeric(df[read_col], errors='coerce')
        if 'AvgMath' not in df.columns:
            math_col = find_col_by_keywords(orig_cols, ['math'])
            if math_col and math_col in df.columns:
                df['AvgMath'] = pd.to_numeric(df[math_col], errors='coerce')
        # Compute AvgScore if we have both now; otherwise create column with NaN
        if 'AvgReading' in df.columns and 'AvgMath' in df.columns:
            df['AvgScore'] = df[['AvgReading', 'AvgMath']].mean(axis=1)
        else:
            df['AvgScore'] = np.nan

    # Compute independent variable: students per teacher
    # Ensure Enrollment and NumTeachers are numeric (already coerced above if present)
    if 'Enrollment' in df.columns and 'NumTeachers' in df.columns:
        # Set to NaN if NumTeachers is zero or missing
        df['StudentTeacherRatio'] = np.where(
            (pd.notna(df['NumTeachers'])) & (df['NumTeachers'] > 0) & (pd.notna(df['Enrollment'])),
            df['Enrollment'] / df['NumTeachers'],
            np.nan
        )
    else:
        df['StudentTeacherRatio'] = np.nan

    # Compute computers per student
    if 'NumComputers' in df.columns and 'Enrollment' in df.columns:
        df['ComputersPerStudent'] = np.where(
            (pd.notna(df['Enrollment'])) & (df['Enrollment'] > 0) & (pd.notna(df['NumComputers'])),
            df['NumComputers'] / df['Enrollment'],
            np.nan
        )
    else:
        df['ComputersPerStudent'] = np.nan

    # Winsorize StudentTeacherRatio to reduce influence of extreme outliers (1st-99th percentile)
    if 'StudentTeacherRatio' in df.columns:
        # compute quantiles only from non-missing values
        non_missing = df['StudentTeacherRatio'].dropna()
        if not non_missing.empty:
            lower = non_missing.quantile(0.01)
            upper = non_missing.quantile(0.99)
            df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=lower, upper=upper)

    # Convert County and GradeSpan to categorical if present
    if 'County' in df.columns:
        df['County'] = df['County'].astype('category')
    if 'GradeSpan' in df.columns:
        df['GradeSpan'] = df['GradeSpan'].astype('category')

    # Ensure all needed columns exist in the returned dataframe; if not, create with NaN so missingness is explicit
    needed = [
        'StudentTeacherRatio', 'AvgScore', 'PercentCalWorks', 'PercentReducedLunch',
        'PercentEnglishLearners', 'ExpenditurePerStudent', 'DistrictIncome',
        'ComputersPerStudent', 'Enrollment', 'County', 'GradeSpan'
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan

    # For stability, remove rows with missing outcome or main predictor
    df = df.dropna(subset=['AvgScore', 'StudentTeacherRatio'])

    # For controls: drop rows missing many key controls (require at least the main socioeconomic controls)
    df = df.dropna(subset=['PercentReducedLunch', 'PercentEnglishLearners', 'ExpenditurePerStudent', 'DistrictIncome'], how='any')

    # Return final dataframe with guaranteed columns present
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model estimating the association between student-teacher ratio and average test score.

    Model specification:
      AvgScore ~ StudentTeacherRatio + controls + categorical county and grade-span fixed effects

    We use heteroskedasticity-robust standard errors (HC3).

    Returns the fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    # Copy to avoid modifying original
    df_model = df.copy()

    # Build formula: include StudentTeacherRatio as main IV and a set of controls.
    # C(County) and C(GradeSpan) add categorical fixed effects.
    formula = (
        'AvgScore ~ StudentTeacherRatio + PercentReducedLunch + PercentCalWorks + '
        'PercentEnglishLearners + ExpenditurePerStudent + DistrictIncome + ComputersPerStudent + Enrollment + '
        'C(County) + C(GradeSpan)'
    )

    # Fit OLS with robust (HC3) standard errors
    model_res = smf.ols(formula=formula, data=df_model).fit(cov_type='HC3')

    # Return the fitted results object (user can call .summary() or inspect params)
    return model_res