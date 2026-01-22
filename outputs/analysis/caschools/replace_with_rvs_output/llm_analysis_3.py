from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/replace_with_rvs_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district dataset to variables needed for modeling.

    Produces the following new columns used by the model:
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of read and math
      - LogStudents: log(total students)
      - ComputersPerStudent: computer / students
      - Expenditure, Income, English, Lunch: copied from original columns
      - county, grades: ensured as categorical

    Rows with missing key fields (students, teachers, read, math) or invalid teacher counts are dropped.
    """
    df = df.copy()

    # Ensure essential columns exist
    required_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'english', 'lunch', 'computer', 'county', 'grades']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing critical fields
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Avoid division by zero for teachers or students
    df = df[df['teachers'] > 0]
    df = df[df['students'] > 0]

    # Construct dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Controls: copy/rename for clarity in modeling
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']
    df['English'] = df['english']
    df['Lunch'] = df['lunch']

    # Log of students (size control) and computers per student
    # Use natural log; add small epsilon to be robust to any extremely small counts (shouldn't be necessary after filtering)
    eps = 1e-6
    df['LogStudents'] = np.log(df['students'] + eps)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Ensure categorical controls are set as object/category types for formula interface
    df['county'] = df['county'].astype('category')
    df['grades'] = df['grades'].astype('category')

    # Keep only columns needed for downstream analysis (but return full df is acceptable; here we ensure required columns exist)
    # This also prevents accidental use of slightly different column names in modeling code.
    model_cols = ['AvgScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'English', 'Lunch', 'LogStudents', 'ComputersPerStudent', 'county', 'grades']
    for c in model_cols:
        if c not in df.columns:
            raise KeyError(f"Expected column {c} not present after transformation")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Estimate the association between student-teacher ratio and district average test performance.

    Model specification:
      AvgScore ~ StudentTeacherRatio + Expenditure + Income + English + Lunch + LogStudents + ComputersPerStudent + C(county) + C(grades)

    Uses heteroskedasticity-robust standard errors (HC3).

    Returns the fitted statsmodels regression results object.
    """
    # local import for formula interface
    import statsmodels.formula.api as smf

    # Ensure transformed columns exist
    needed = ['AvgScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'English', 'Lunch', 'LogStudents', 'ComputersPerStudent', 'county', 'grades']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    formula = (
        'AvgScore ~ StudentTeacherRatio + Expenditure + Income + English + Lunch '
        '+ LogStudents + ComputersPerStudent + C(county) + C(grades)'
    )

    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted results object so the caller can inspect summary, params, diagnostics, etc.
    return model


