from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/replace_with_rvs_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into a dataset suitable for OLS modeling.

    Creates the following derived columns used in the model:
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of 'read' and 'math'
      - ComputersPerStudent: computer / students
      - LogStudents: natural log of students
      - Grades_KK08: indicator (1 if grades == 'KK-08', 0 otherwise)

    Drops rows with missing values in any columns required by the model.
    """
    # work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'read', 'math']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived variables
    # Avoid division by zero / invalid teachers values
    df['teachers'] = df['teachers'].replace({0: np.nan})

    # Student-teacher ratio: students per teacher
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Average academic score: mean of read and math
    # If one of read/math is missing, mean will be NaN -> we will drop later
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log of students (for scale control). Protect against nonpositive values
    df['LogStudents'] = np.where(df['students'] > 0, np.log(df['students']), np.nan)

    # Grades indicator: 1 if KK-08, 0 if KK-06 (or other treat as 0). Create column only if 'grades' exists
    if 'grades' in df.columns:
        df['Grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)
    else:
        df['Grades_KK08'] = 0

    # Keep county as-is (used as fixed effect in the model). Ensure it is treated as categorical
    if 'county' in df.columns:
        df['county'] = df['county'].astype(str)

    # List all columns needed for modeling
    required_cols = [
        'StudentTeacherRatio', 'AvgScore', 'expenditure', 'income', 'english', 'lunch', 'calworks',
        'ComputersPerStudent', 'LogStudents', 'Grades_KK08', 'county'
    ]

    # Drop rows with missing values in any required column
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Remove infinite values if any
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=existing_required)

    # Final dataframe returned for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for several district-level covariates
    and county fixed effects. Returns the model results with robust (HC3) standard errors.

    Model specification (formula):
      AvgScore ~ StudentTeacherRatio + expenditure + income + english + lunch + calworks
                 + ComputersPerStudent + LogStudents + Grades_KK08 + C(county)

    """
    import statsmodels.formula.api as smf

    # Ensure the required columns exist
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + english + lunch + calworks '
        '+ ComputersPerStudent + LogStudents + Grades_KK08 + C(county)'
    )

    # Fit OLS
    model = smf.ols(formula, data=df).fit()

    # Obtain robust standard errors (HC3)
    results_robust = model.get_robustcov_results(cov_type='HC3')

    # Print brief summary to console (optional)
    print(results_robust.summary())

    return results_robust


