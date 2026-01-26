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
    Transform the raw district-level dataset to create the variables required for modeling.

    Produces the following derived columns used in the model:
      - AvgTestScore: mean of 'read' and 'math'
      - StudentTeacherRatio: students / teachers
      - ComputersPerStudent: computer / students
      - LogStudents: natural log of students
      - is_KK08: indicator for grades == 'KK-08'
      - county: coerced to string for creating dummies later

    Also drops rows with missing or invalid core values (students, teachers, read, math).
    """
    df = df.copy()

    # Core columns required for the analysis
    required_core = ['students', 'teachers', 'read', 'math']
    # Drop rows missing core values
    df = df.dropna(subset=required_core)

    # Remove rows with non-positive counts that would break ratios/logs
    df = df[(df['students'] > 0) & (df['teachers'] > 0)]

    # Dependent variable: average of reading and math scores
    df['AvgTestScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: students per teacher
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Computers per student (handle zeros for students above)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log-transformed enrollment size
    df['LogStudents'] = np.log(df['students'])

    # Grade-span indicator
    df['is_KK08'] = (df['grades'].astype(str).str.strip() == 'KK-08').astype(int)

    # Ensure county is a string (for later dummy creation)
    df['county'] = df['county'].astype(str)

    # Replace infinite values (just in case) and drop resulting missing
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Many controls are used in the model; drop rows missing any of them
    controls_needed = ['expenditure', 'income', 'calworks', 'lunch', 'english',
                       'StudentTeacherRatio', 'AvgTestScore', 'ComputersPerStudent', 'LogStudents', 'is_KK08', 'county']
    df = df.dropna(subset=controls_needed)

    # Final copy returned
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit an OLS regression of average test score on student-teacher ratio controlling for
    district characteristics and county fixed effects. Robust (HC3) standard errors
    are used to make inference more robust to heteroskedasticity.

    Model specification (linear):
      AvgTestScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch
                     + english + ComputersPerStudent + LogStudents + is_KK08 + county FE

    Returns:
      statsmodels regression results object (with HC3 robust SEs)
    """
    # Work on a copy
    df = df.copy()

    # Create county dummies (drop_first to avoid perfect multicollinearity)
    county_dummies = pd.get_dummies(df['county'], prefix='county', drop_first=True)

    # Select covariates
    X_vars = [
        'StudentTeacherRatio',
        'expenditure',
        'income',
        'calworks',
        'lunch',
        'english',
        'ComputersPerStudent',
        'LogStudents',
        'is_KK08'
    ]

    # Ensure all required columns are present
    X = pd.concat([df[X_vars].reset_index(drop=True), county_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)

    # Outcome
    y = df['AvgTestScore']

    # Fit OLS with robust standard errors (HC3)
    ols_res = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted model object so the caller can inspect summary, params, etc.
    return ols_res


