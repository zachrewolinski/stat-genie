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
    Transform the raw dataset into a dataframe ready for modeling.

    Returns a dataframe containing at least the following columns used in the model:
      - AvgScore
      - StudentTeacherRatio
      - Income
      - PercentCalWorks
      - PercentLunch
      - PercentEnglishLearners
      - ExpenditurePerStudent
      - ComputersPerStudent
      - Grades_KK08
      - county
    """
    df = df.copy()

    # Ensure key numeric columns exist
    required_cols = ['read', 'math', 'students', 'teachers', 'computer', 'income', 'calworks', 'lunch', 'english', 'expenditure', 'grades', 'county']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows missing the primary score or size/teacher information
    df = df.dropna(subset=['read', 'math', 'students', 'teachers'])

    # Dependent variable: average of reading and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio (students per teacher)
    # Protect against division by zero
    df['StudentTeacherRatio'] = np.where(df['teachers'] == 0, np.nan, df['students'] / df['teachers'])

    # Control: computers per student
    df['ComputersPerStudent'] = np.where(df['students'] == 0, np.nan, df['computer'] / df['students'])

    # Map and rename existing columns into clearer control variable names used in model
    df['Income'] = df['income']
    df['PercentCalWorks'] = df['calworks']
    df['PercentLunch'] = df['lunch']
    df['PercentEnglishLearners'] = df['english']
    df['ExpenditurePerStudent'] = df['expenditure']

    # Create a binary indicator for grade-span KK-08 (1 if KK-08, 0 otherwise)
    # Ensure grades is string-like
    df['Grades_KK08'] = df['grades'].astype(str).str.strip().str.upper().eq('KK-08').astype(int)

    # Ensure county is a string/categorical variable
    df['county'] = df['county'].astype(str)

    # Final list of columns we will require for the model
    model_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'Income',
        'PercentCalWorks',
        'PercentLunch',
        'PercentEnglishLearners',
        'ExpenditurePerStudent',
        'ComputersPerStudent',
        'Grades_KK08',
        'county'
    ]

    # Drop rows with missing values in any column used in the model
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio with controls and county fixed effects.

    Model formula used:
      AvgScore ~ StudentTeacherRatio + Income + PercentCalWorks + PercentLunch
                 + PercentEnglishLearners + ExpenditurePerStudent + ComputersPerStudent
                 + Grades_KK08 + C(county)

    Returns the fitted results object with robust (HC3) standard errors.
    """
    import statsmodels.formula.api as smf

    # Copy dataframe to avoid side-effects
    df = df.copy()

    # Ensure required columns are present
    formula = (
        'AvgScore ~ StudentTeacherRatio + Income + PercentCalWorks + PercentLunch '
        '+ PercentEnglishLearners + ExpenditurePerStudent + ComputersPerStudent '
        '+ Grades_KK08 + C(county)'
    )

    # Fit OLS
    model_fit = smf.ols(formula, data=df).fit()

    # Compute robust standard errors (HC3) and return robust results
    robust_results = model_fit.get_robustcov_results(cov_type='HC3')

    return robust_results


