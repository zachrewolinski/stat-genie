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
    Transform the raw district-level dataframe to create the analysis-ready columns:
      - StudentTeacherRatio: students / teachers (NaN when teachers <= 0)
      - AvgTestScore: mean of 'read' and 'math'
      - ComputersPerStudent: computer / students (0 if computer==0 and students>0)
      - LogStudents: natural log of students

    Drops rows with missing essential input fields (students, teachers, read, math).
    Returns the dataframe with new columns added.
    """
    df = df.copy()

    # Ensure numeric columns exist and coerce to numeric where appropriate
    numeric_cols = ['students', 'teachers', 'computer', 'read', 'math', 'expenditure', 'income', 'calworks', 'lunch', 'english']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing essential inputs for computing ratio and outcome
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Avoid division by zero or negative teacher counts
    df.loc[df['teachers'] <= 0, 'teachers'] = np.nan

    # Student-teacher ratio: students per teacher
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of read and math scores
    df['AvgTestScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student (handle zero or missing students)
    df['ComputersPerStudent'] = np.nan
    valid_students = df['students'] > 0
    df.loc[valid_students, 'ComputersPerStudent'] = df.loc[valid_students, 'computer'] / df.loc[valid_students, 'students']

    # Log of students to control for district size (handle zeros and negatives safely)
    df['LogStudents'] = np.nan
    df.loc[valid_students, 'LogStudents'] = np.log(df.loc[valid_students, 'students'])

    # (Optional) If control variables have non-numeric issues, coerce them earlier; here just keep as-is

    # Final cleanup: drop rows with NaN in the primary variables used in the model
    required_for_model = [
        'AvgTestScore', 'StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'LogStudents'
    ]
    # Keep rows where at least the primary DV and IV and core controls exist. We don't force all controls if user wants fewer,
    # but for the planned model we'll drop rows missing any of these columns.
    df = df.dropna(subset=required_for_model)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating the association between StudentTeacherRatio and AvgTestScore,
    controlling for district characteristics. Uses robust (HC3) standard errors.

    Model specification:
      AvgTestScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english + ComputersPerStudent + LogStudents

    Returns the fitted statsmodels results object.
    """
    # Work on a copy
    data = df.copy()

    # Ensure required columns exist
    required = ['AvgTestScore', 'StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'LogStudents']
    missing = [c for c in required if c not in data.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop any remaining rows with missing values in model columns
    data = data.dropna(subset=required)

    # Define outcome and predictors
    y = data['AvgTestScore']
    X = data[['StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'LogStudents']]

    # Add intercept
    X = sm.add_constant(X)

    # Fit OLS with robust standard errors (HC3)
    model_res = sm.OLS(y, X).fit(cov_type='HC3')

    # Print brief summary to console (user can use the returned object for full inspection)
    print(model_res.summary())

    return model_res


