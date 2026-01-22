from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe to prepare variables for modeling.

    Outputs (columns guaranteed if present in input):
    - AcademicScore: mean of 'read' and 'math'
    - StudentTeacherRatio: students / teachers (capped at 99th percentile to reduce influence of extreme outliers)
    - LogStudents: log1p(students)
    - StdStudentTeacherRatio, StdAcademicScore: standardized versions (z-scores)
    - plus the original control columns (expenditure, income, calworks, lunch, english, county, grades)
    """

    df = df.copy()

    # Ensure numeric conversion for key numeric columns; coerce errors to NaN
    num_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'calworks', 'lunch', 'english']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows that lack the critical variables needed to compute DV/IV
    required = [c for c in ['students', 'teachers', 'read', 'math'] if c in df.columns]
    df = df.dropna(subset=required)

    # Remove rows with non-positive teacher counts (can't compute ratio)
    if 'teachers' in df.columns:
        df = df[df['teachers'] > 0]

    # Compute dependent variable: average of read and math
    df['AcademicScore'] = df[['read', 'math']].mean(axis=1)

    # Compute independent variable: students per teacher
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Trim extreme outliers in the ratio by capping at the 99th percentile
    if df['StudentTeacherRatio'].notna().any():
        p99 = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(upper=p99)

    # Auxiliary transforms
    df['LogStudents'] = np.log1p(df['students'])

    # Standardized versions for interpretation (use population std to avoid ddof mismatches)
    df['StdStudentTeacherRatio'] = (df['StudentTeacherRatio'] - df['StudentTeacherRatio'].mean()) / (df['StudentTeacherRatio'].std(ddof=0) if df['StudentTeacherRatio'].std(ddof=0) != 0 else 1)
    df['StdAcademicScore'] = (df['AcademicScore'] - df['AcademicScore'].mean()) / (df['AcademicScore'].std(ddof=0) if df['AcademicScore'].std(ddof=0) != 0 else 1)

    # Ensure categorical variables are treated as categories
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')

    # Keep only columns necessary for modeling and diagnostics (if they exist)
    keep = [
        'AcademicScore', 'StudentTeacherRatio', 'StdStudentTeacherRatio', 'StdAcademicScore',
        'expenditure', 'income', 'calworks', 'lunch', 'english', 'students', 'LogStudents',
        'county', 'grades', 'teachers'
    ]
    keep_present = [c for c in keep if c in df.columns]
    df = df[keep_present]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of academic performance on the student-teacher ratio and controls.

    Model specification:
    AcademicScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english + np.log(students) + C(county) + C(grades)

    Returns the fitted statsmodels RegressionResults object (with robust standard errors, HC3).
    """

    import statsmodels.formula.api as smf
    # Copy input to avoid side-effects
    data = df.copy()

    # Ensure 'students' is numeric for the log term
    if 'students' in data.columns:
        data['students'] = pd.to_numeric(data['students'], errors='coerce')

    # Drop rows with any remaining missing values in variables used by the formula
    formula = 'AcademicScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english + np.log(students) + C(county) + C(grades)'
    # Identify variables used in the formula to drop NA rows
    vars_needed = ['AcademicScore', 'StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'students', 'county', 'grades']
    vars_present = [v for v in vars_needed if v in data.columns]
    data = data.dropna(subset=vars_present)

    # Fit OLS with robust standard errors (HC3)
    model = smf.ols(formula, data=data).fit(cov_type='HC3')

    # Return the fitted model object so the caller can inspect summary, params, etc.
    return model


