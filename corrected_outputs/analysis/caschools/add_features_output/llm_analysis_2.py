from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.
    Produces the following new columns used in the model:
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of 'read' and 'math'
      - ComputersPerStudent: computer / students
      - PctLunch: alias for 'lunch'
      - PctCalworks: alias for 'calworks'
      - PctEnglishLearners: alias for 'english'
      - Expenditure: alias for 'expenditure'
      - Income: alias for 'income'
      - LogStudents: log(students)
      - Grades_KK08: indicator for grades == 'KK-08' (1) else 0

    The function also drops rows with missing or invalid values in the required columns.
    """
    df = df.copy()

    # Columns required for the transformations and model
    required_cols = [
        'students', 'teachers', 'read', 'math', 'expenditure', 'income',
        'lunch', 'calworks', 'english', 'computer', 'grades'
    ]

    # Ensure numeric columns are numeric where appropriate
    numeric_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'lunch', 'calworks', 'english', 'computer']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing any required information
    df = df.dropna(subset=required_cols)

    # Remove rows with non-positive students or teachers
    df = df[(df['students'] > 0) & (df['teachers'] > 0)]

    # Create Student-Teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Create average score (dependent variable)
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Aliases for controls (rename / copy into model columns)
    df['PctLunch'] = df['lunch']
    df['PctCalworks'] = df['calworks']
    df['PctEnglishLearners'] = df['english']
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']

    # District size (log)
    # Guard against zeros (we already filtered students > 0)
    df['LogStudents'] = np.log(df['students'])

    # Binary indicator for grade span (KK-08 vs KK-06). If other category names appear, treat them as 0 unless explicitly 'KK-08'
    df['Grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)

    # Final drop: ensure no missing values remain in model columns
    model_cols = [
        'AvgScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'PctLunch',
        'PctCalworks', 'PctEnglishLearners', 'ComputersPerStudent', 'LogStudents', 'Grades_KK08'
    ]
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits an OLS regression of AvgScore on StudentTeacherRatio and controls.
    Uses heteroskedasticity-robust standard errors (HC3).

    Model specification:
      AvgScore = beta0 + beta1*StudentTeacherRatio + beta2*Expenditure + beta3*Income
                 + beta4*PctLunch + beta5*PctCalworks + beta6*PctEnglishLearners
                 + beta7*ComputersPerStudent + beta8*LogStudents + beta9*Grades_KK08 + error

    Returns the fitted statsmodels regression results object (with robust SEs).
    """
    # Select only rows that have the model columns
    model_cols = [
        'AvgScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'PctLunch',
        'PctCalworks', 'PctEnglishLearners', 'ComputersPerStudent', 'LogStudents', 'Grades_KK08'
    ]
    df_model = df.dropna(subset=model_cols).copy()

    # Design matrix X and outcome y
    X = df_model[[
        'StudentTeacherRatio', 'Expenditure', 'Income', 'PctLunch', 'PctCalworks',
        'PctEnglishLearners', 'ComputersPerStudent', 'LogStudents', 'Grades_KK08'
    ]]
    X = sm.add_constant(X)
    y = df_model['AvgScore']

    # Fit OLS with robust (HC3) standard errors
    results = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted results object. The caller can inspect results.summary(), results.params, etc.
    return results


