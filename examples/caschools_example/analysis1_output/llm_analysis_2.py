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
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric where appropriate
    numeric_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'read', 'math']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Remove rows with missing critical values: students, teachers, read, math
    req = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=req)

    # Remove unrealistic teacher counts (<= 0) to avoid division by zero
    df = df[df['teachers'] > 0]

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Control: computers per student (avoid divide-by-zero; students > 0 ensured above)
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Log enrollment (use natural log); guard against nonpositive students
    df['LogStudents'] = np.log(df['students'].astype(float))

    # Grade-span dummy: 1 if KK-08, 0 otherwise (including KK-06)
    if 'grades' in df.columns:
        df['Grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)
    else:
        df['Grades_KK08'] = 0

    # Keep columns needed for modeling plus originals for context
    needed_cols = [
        'AvgScore', 'StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english',
        'ComputerPerStudent', 'LogStudents', 'Grades_KK08'
    ]

    # If any control columns are missing in the input dataset, create them as NaN to keep consistent schema
    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows with missing values in any of the model columns
    df = df.dropna(subset=needed_cols)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Build design matrix
    model_cols = [
        'StudentTeacherRatio',
        'expenditure',
        'income',
        'calworks',
        'lunch',
        'english',
        'ComputerPerStudent',
        'LogStudents',
        'Grades_KK08'
    ]

    X = df[model_cols].astype(float)
    X = sm.add_constant(X)
    y = df['AvgScore'].astype(float)

    # Fit OLS with robust (HC1) standard errors
    ols_model = sm.OLS(y, X).fit(cov_type='HC1')

    # Return the fitted results object. Caller can print summary via print(ols_model.summary())
    return ols_model


