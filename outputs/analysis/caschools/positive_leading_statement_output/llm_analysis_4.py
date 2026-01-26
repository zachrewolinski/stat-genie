from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/positive_leading_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare variables for analysis.
    Creates:
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of read and math
      - ComputersPerStudent: computer / students
    Drops rows with invalid or missing values for core variables and trims extreme StudentTeacherRatio outliers (1st/99th percentile).
    Returns the transformed dataframe with the columns used in the model.
    """
    df = df.copy()

    # Ensure relevant numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with impossible teacher/student values and core missings
    if 'teachers' in df.columns:
        df = df[df['teachers'] > 0]
    if 'students' in df.columns:
        df = df[df['students'] > 0]

    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Construct variables
    df['StudentTeacherRatio'] = df['students'] / df['teachers']
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student; if computer or students missing this will be NaN
    if 'computer' in df.columns:
        df['ComputersPerStudent'] = df['computer'] / df['students']
    else:
        df['ComputersPerStudent'] = np.nan

    # Replace infinite values and fill obvious NAs for computers per student with 0 (no computers)
    df['ComputersPerStudent'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['ComputersPerStudent'] = df['ComputersPerStudent'].fillna(0)

    # Trim extreme StudentTeacherRatio outliers to reduce influence (keep between 1st and 99th percentile)
    if 'StudentTeacherRatio' in df.columns:
        q_low = df['StudentTeacherRatio'].quantile(0.01)
        q_high = df['StudentTeacherRatio'].quantile(0.99)
        df = df[(df['StudentTeacherRatio'] >= q_low) & (df['StudentTeacherRatio'] <= q_high)]

    # Keep only columns that will be used by the model plus identifiers
    keep_cols = [c for c in df.columns if c in (
        ['StudentTeacherRatio', 'AvgScore', 'ComputersPerStudent', 'students', 'teachers', 'read', 'math',
         'expenditure', 'income', 'calworks', 'lunch', 'english', 'grades', 'county', 'district', 'school', 'rownames']
    )]
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating association between student-teacher ratio and average test scores.
    Controls: expenditure, income, calworks, lunch, english, ComputersPerStudent, grades (categorical), county fixed effects.
    Uses heteroskedasticity-robust (HC3) standard errors.

    Returns the fitted statsmodels regression result object.
    """
    import statsmodels.formula.api as smf

    # Ensure the columns exist
    required = ['AvgScore', 'StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'grades', 'county']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula with categorical controls for grades and county (fixed effects)
    # Primary coefficient of interest: StudentTeacherRatio
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english + ComputersPerStudent + '
        'C(grades) + C(county)'
    )

    # Fit OLS with robust standard errors (HC3)
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return fitted model (call .summary() externally if textual output is desired)
    return model


