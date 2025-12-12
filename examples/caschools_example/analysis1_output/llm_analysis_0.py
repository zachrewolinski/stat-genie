from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/caschools/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe to produce the variables needed for modeling.

    Produces:
    - StudentTeacherRatio: students / teachers
    - AvgScore: mean of 'read' and 'math'
    - ComputersPerStudent: computer / students
    - LogStudents: natural log of students
    - Grades_KK08: dummy (1 if grades == 'KK-08', else 0)
    - Ensures 'county' is string (categorical)

    Drops rows with missing or invalid values in core columns.
    """
    df = df.copy()

    # Drop rows missing core numeric inputs needed to compute key variables
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove rows with non-positive teacher counts to avoid division by zero
    df = df[df['teachers'] > 0]

    # Compute independent variable: students per teacher
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Compute dependent variable: average score across reading and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student: if computer or students are missing this will be NaN (kept for later drop)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log of total students (replace zeros defensively, though zeros were removed earlier)
    df['LogStudents'] = np.log(df['students'].replace(0, np.nan))

    # Grade-span indicator: 1 if 'KK-08', 0 otherwise
    # Ensure grades is string before comparison
    df['Grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)

    # Ensure county is string for later dummy creation
    df['county'] = df['county'].astype(str)

    # Return transformed dataframe (retain all original columns plus derived ones)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model to estimate the association between student-teacher ratio and average test score.

    Model specification:
      AvgScore_i = beta0 + beta1 * StudentTeacherRatio_i + beta2 * expenditure_i + beta3 * income_i
                   + beta4 * lunch_i + beta5 * english_i + beta6 * ComputersPerStudent_i
                   + beta7 * LogStudents_i + beta8 * Grades_KK08_i + county fixed effects + error_i

    Uses heteroskedasticity-robust (HC3) standard errors.

    Returns the fitted statsmodels RegressionResults object.
    """
    df = df.copy()

    # Columns required for the model
    required_cols = [
        'StudentTeacherRatio', 'expenditure', 'income', 'lunch', 'english',
        'ComputersPerStudent', 'LogStudents', 'Grades_KK08', 'AvgScore', 'county'
    ]

    # Drop rows with missing values in any required column
    df_model = df.dropna(subset=required_cols)

    # Create county dummy variables (drop_first to avoid perfect multicollinearity)
    county_dummies = pd.get_dummies(df_model['county'], prefix='county', drop_first=True)

    # Construct feature matrix X
    X_cols = [
        'StudentTeacherRatio', 'expenditure', 'income', 'lunch', 'english',
        'ComputersPerStudent', 'LogStudents', 'Grades_KK08'
    ]

    X = df_model[X_cols].join(county_dummies)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Outcome
    y = df_model['AvgScore']

    # Fit OLS with robust (HC3) standard errors
    results = sm.OLS(y, X).fit(cov_type='HC3')

    return results


