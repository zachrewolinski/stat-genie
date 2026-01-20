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
    # Work on a copy
    df = df.copy()

    # Ensure numeric types for key numeric columns (coerce errors to NaN)
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'english', 'lunch']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core vars needed to compute ratio and outcome
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove invalid teacher counts (avoid division by zero or negative counts)
    df = df[df['teachers'] > 0]

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Additional derived controls
    # computers per student (may be small values); coerce students>0 already
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Rename / copy columns into model-oriented names
    df['PerStudentExpenditure'] = df['expenditure']
    df['PctEnglishLearners'] = df['english']
    df['PctReducedLunch'] = df['lunch']

    # Log of students to capture size effects
    # Add small positive constant if any schools have 0 students (shouldn't happen given data), but guard anyway
    df['LogStudents'] = np.log(df['students'].clip(lower=1))

    # Grade-span indicator: 1 if KK-08, 0 if KK-06 or other
    if 'grades' in df.columns:
        df['KK08'] = (df['grades'] == 'KK-08').astype(int)
    else:
        df['KK08'] = 0

    # Final model columns to ensure no missing values
    model_cols = [
        'StudentTeacherRatio', 'PerStudentExpenditure', 'income', 'PctEnglishLearners',
        'PctReducedLunch', 'ComputerPerStudent', 'LogStudents', 'KK08', 'AvgScore'
    ]
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Prepare design matrix
    model_df = df.copy()
    X_cols = [
        'StudentTeacherRatio',
        'PerStudentExpenditure',
        'income',
        'PctEnglishLearners',
        'PctReducedLunch',
        'ComputerPerStudent',
        'LogStudents',
        'KK08'
    ]

    # Ensure all predictors present
    X = model_df[X_cols]
    X = sm.add_constant(X)
    y = model_df['AvgScore']

    # Fit linear OLS with robust (heteroskedasticity-consistent) standard errors
    ols_model = sm.OLS(y, X).fit(cov_type='HC3')

    # Print and return results object
    print(ols_model.summary())
    return ols_model


