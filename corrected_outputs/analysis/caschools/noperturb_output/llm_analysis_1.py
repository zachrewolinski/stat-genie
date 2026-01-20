from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/noperturb_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into the analysis-ready dataframe.

    Produces the following new columns required for modeling:
    - avg_score: mean of 'read' and 'math'
    - student_teacher_ratio: students / teachers
    - computers_per_student: computer / students
    - grades_KK08: binary indicator (1 if grades == 'KK-08', else 0)
    - log_students: log of students (for size control)

    Drops rows missing essential fields (students, teachers, read, math).
    """
    df = df.copy()

    # Drop rows with missing essential variables
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Dependent variable: average of reading and math scores
    df['avg_score'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio (students per teacher)
    # Ensure teachers non-zero (we dropped missing teachers above); guard against zero
    df['student_teacher_ratio'] = df['students'] / df['teachers'].replace({0: np.nan})

    # Control: computers per student
    df['computers_per_student'] = df['computer'] / df['students'].replace({0: np.nan})

    # Control: grades indicator KK-08 (vs KK-06). Cast to string to be robust.
    df['grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)

    # Control: log of student enrollment (use natural log); replace zeros with NaN first
    df['log_students'] = np.log(df['students'].replace({0: np.nan}))

    # Keep the new columns and relevant original controls in the dataframe (do not remove other columns)
    # This ensures downstream code can still access original variables if needed.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and average test scores.

    Model specification (primary):
      avg_score ~ student_teacher_ratio + expenditure + lunch + english + income + computers_per_student + grades_KK08 + log_students

    Uses heteroskedasticity-robust standard errors (HC3).

    Returns the fitted statsmodels results object.
    """
    # Select columns required for modeling and drop rows with any missing values among them
    required_cols = [
        'avg_score',
        'student_teacher_ratio',
        'expenditure',
        'lunch',
        'english',
        'income',
        'computers_per_student',
        'grades_KK08',
        'log_students'
    ]

    df_model = df.copy()
    df_model = df_model.dropna(subset=required_cols)

    # Design matrix
    X = df_model[[
        'student_teacher_ratio',
        'expenditure',
        'lunch',
        'english',
        'income',
        'computers_per_student',
        'grades_KK08',
        'log_students'
    ]]
    X = sm.add_constant(X)
    y = df_model['avg_score']

    # Fit OLS with robust standard errors (HC3)
    results = sm.OLS(y, X).fit(cov_type='HC3')

    return results


