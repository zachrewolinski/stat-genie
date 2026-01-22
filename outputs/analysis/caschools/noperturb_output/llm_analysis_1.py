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
    Transform the original dataframe to produce the columns used in modeling.

    Output columns (added/modified):
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of 'read' and 'math'
      - ComputerPerStudent: computer / students
      - LogStudents: log1p(students)
      - county, grades converted to categorical

    The function drops rows with missing core variables (students, teachers, read, math)
    and drops rows where teachers <= 0 to avoid division-by-zero.
    """
    df = df.copy()

    # Coerce core numeric columns to numeric (safe conversion)
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'english', 'lunch', 'calworks']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing core variables required to compute the IV and DV
    required_core = ['students', 'teachers', 'read', 'math']
    present_required = [c for c in required_core if c in df.columns]
    df = df.dropna(subset=present_required)

    # Remove rows with non-positive teacher counts to avoid division by zero
    df = df[df['teachers'] > 0]

    # Student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student
    if 'computer' in df.columns:
        # avoid division by zero (students>0 preserved above)
        df['ComputerPerStudent'] = df['computer'] / df['students']
    else:
        df['ComputerPerStudent'] = np.nan

    # Log of student count to control for district size (use log1p to be safe)
    df['LogStudents'] = np.log1p(df['students'])

    # Cast categorical controls to category dtype for modeling convenience
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')

    # Keep original numeric control columns as-is (expenditure, income, english, lunch, calworks)
    # Note: rows with missing control vars will be handled by the modeling function (dropped in model-ready subset)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Estimate the association between student-teacher ratio and district average academic performance.

    Model specification (linear OLS with county and grade-span fixed effects):
      AvgScore ~ StudentTeacherRatio + expenditure + income + english + lunch + calworks
                 + ComputerPerStudent + LogStudents + C(county) + C(grades)

    Robust (HC3) standard errors are used.

    Returns the fitted statsmodels results object with robust covariance.
    """
    import statsmodels.formula.api as smf

    # Work on a copy; assume transform() has been applied already
    df = df.copy()

    # List of variables used in the model; only keep rows with non-missing values here
    model_vars = [
        'AvgScore', 'StudentTeacherRatio', 'expenditure', 'income', 'english', 'lunch', 'calworks',
        'ComputerPerStudent', 'LogStudents', 'county', 'grades'
    ]
    present_vars = [v for v in model_vars if v in df.columns]
    df_model = df.dropna(subset=present_vars)

    # Construct formula
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + english + lunch + calworks '
        '+ ComputerPerStudent + LogStudents + C(county) + C(grades)'
    )

    # Fit OLS
    mod = smf.ols(formula=formula, data=df_model).fit()

    # Convert to robust covariance (HC3) for heteroskedasticity-robust SEs
    robust_res = mod.get_robustcov_results(cov_type='HC3')

    # Print brief summary (user can inspect robust_res.summary())
    print(robust_res.summary())

    return robust_res


