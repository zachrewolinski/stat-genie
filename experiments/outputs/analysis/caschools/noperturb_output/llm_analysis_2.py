from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/noperturb_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original district-level dataframe to create the analysis-ready dataframe.

    Produces the following columns required by the model:
      - StudentsPerTeacher: students / teachers
      - AvgScore: mean of 'read' and 'math'
      - ComputersPerStudent: computer / students
      - Expenditure, English, Lunch, Income, Students, Grades, County (copied/converted)

    Rows with missing or invalid critical values (e.g., teachers <= 0) are removed.
    """
    df = df.copy()

    # Ensure required input columns exist (if not, this will raise a KeyError)
    required = ['students', 'teachers', 'read', 'math', 'expenditure', 'english', 'lunch', 'income', 'computer', 'grades', 'county']

    # Drop rows missing critical variables for IV and DV
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove impossible or degenerate values
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Compute independent variable: students per teacher
    df['StudentsPerTeacher'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Controls and derived resource variable
    # ComputersPerStudent: if 'computer' is total number of computers, divide by students
    # We allow zero computers but drop rows where students==0 above
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Copy other controls into cleanly-named columns used in the model
    df['Expenditure'] = df['expenditure']
    df['English'] = df['english']
    df['Lunch'] = df['lunch']
    df['Income'] = df['income']
    df['Students'] = df['students']

    # Convert categorical controls to string type for downstream formula handling
    df['Grades'] = df['grades'].astype(str)
    df['County'] = df['county'].astype(str)

    # Drop rows with missing control values (we keep only complete cases for this regression)
    df = df.dropna(subset=['Expenditure', 'English', 'Lunch', 'Income', 'ComputersPerStudent', 'Grades', 'County'])

    # Return only the columns required for modeling (keeps namespace small and explicit)
    return df[['StudentsPerTeacher', 'AvgScore', 'Expenditure', 'English', 'Lunch', 'Income', 'ComputersPerStudent', 'Students', 'Grades', 'County']]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Estimate the association between student-teacher ratio and average academic performance.

    Model specification: OLS regression with district-level controls and county & grade-span fixed effects.
    Robust (HC3) standard errors are used to mitigate heteroskedasticity.

    Formula:
      AvgScore ~ StudentsPerTeacher + Expenditure + English + Lunch + Income + ComputersPerStudent + Students + C(County) + C(Grades)

    Returns the fitted statsmodels results object (and prints a summary).
    """
    import statsmodels.formula.api as smf

    # Ensure the dataframe contains the expected columns
    required = ['StudentsPerTeacher', 'AvgScore', 'Expenditure', 'English', 'Lunch', 'Income', 'ComputersPerStudent', 'Students', 'County', 'Grades']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula with categorical fixed effects for County and Grades
    formula = (
        'AvgScore ~ StudentsPerTeacher + Expenditure + English + Lunch + Income + '
        'ComputersPerStudent + Students + C(County) + C(Grades)'
    )

    # Fit OLS with heteroskedasticity-robust standard errors (HC3)
    results = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Print a concise summary; return results for programmatic access
    print(results.summary())
    return results


