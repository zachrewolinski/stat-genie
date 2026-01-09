from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/caschools/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw district-level data into analysis-ready dataframe.

    Creates the following new columns used in the model:
      - StudentTeacherRatio: calworks / teachers (total enrollment divided by number of teachers)
      - AvgTestScore: mean of 'grades' and 'rownames' (district average test performance)
      - ExpenditurePerStudent: numeric copy of 'expenditure'
      - PctReducedLunch: numeric copy of 'math' (as provided by dataset schema)
      - PctCalWorks: numeric copy of 'income' (percent qualifying for assistance)
      - PctEnglishLearners: numeric copy of 'district' (percent English learners)
      - Computers: numeric copy of 'computer'
      - Is_KK08: dummy for school == 'KK-08'

    Drops rows missing the dependent variable or the main independent variable.
    """
    df = df.copy()

    # Coerce numeric columns to numeric types where appropriate
    numeric_cols = ['calworks', 'teachers', 'grades', 'rownames', 'expenditure', 'math', 'income', 'district', 'computer']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create clearly named numeric copies for enrollment and teachers
    if 'calworks' in df.columns:
        df['TotalEnrollment'] = df['calworks']
    else:
        df['TotalEnrollment'] = np.nan

    if 'teachers' in df.columns:
        df['NumTeachers'] = df['teachers']
    else:
        df['NumTeachers'] = np.nan

    # Avoid division by zero
    df.loc[df['NumTeachers'] == 0, 'NumTeachers'] = np.nan

    # Student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['TotalEnrollment'] / df['NumTeachers']

    # Dependent variable: average test score. Use mean of 'grades' and 'rownames' when available.
    # If only one is present, mean will use that one (pandas mean skips NaN by default when axis=1).
    present_score_cols = [c for c in ['grades', 'rownames'] if c in df.columns]
    if len(present_score_cols) == 0:
        # no test score columns available; create column of NaN (will be dropped below)
        df['AvgTestScore'] = np.nan
    else:
        df['AvgTestScore'] = df[present_score_cols].mean(axis=1)

    # Controls: create clean numeric columns with descriptive names
    df['ExpenditurePerStudent'] = df['expenditure'] if 'expenditure' in df.columns else np.nan
    df['PctReducedLunch'] = df['math'] if 'math' in df.columns else np.nan
    df['PctCalWorks'] = df['income'] if 'income' in df.columns else np.nan
    df['PctEnglishLearners'] = df['district'] if 'district' in df.columns else np.nan
    df['Computers'] = df['computer'] if 'computer' in df.columns else np.nan

    # Grade-span dummy: create a binary indicator for KK-08 if 'school' column exists
    if 'school' in df.columns:
        # Ensure string comparison is robust
        df['Is_KK08'] = df['school'].astype(str).fillna('').apply(lambda x: 1 if x.strip() == 'KK-08' else 0)
    else:
        df['Is_KK08'] = 0

    # Replace infinities and drop rows that lack the main variables needed for analysis
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows missing the dependent variable or the main independent variable
    df = df.dropna(subset=['AvgTestScore', 'StudentTeacherRatio'])

    # Optionally: drop rows with missing control information is not strictly necessary for OLS with missing='drop'
    # but to keep a consistent sample we keep rows with at least StudentTeacherRatio and AvgTestScore.

    # Final: ensure the final columns used in modeling exist
    final_cols = ['StudentTeacherRatio', 'AvgTestScore', 'ExpenditurePerStudent', 'PctReducedLunch', 'PctCalWorks', 'PctEnglishLearners', 'Computers', 'Is_KK08']
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of average test score on student-teacher ratio with controls.

    Model specification (linear):
      AvgTestScore = beta0 + beta1 * StudentTeacherRatio + beta2 * ExpenditurePerStudent
                     + beta3 * PctReducedLunch + beta4 * PctCalWorks + beta5 * PctEnglishLearners
                     + beta6 * Computers + beta7 * Is_KK08 + error

    Uses heteroskedasticity-robust standard errors (HC3).
    Returns the fitted statsmodels regression results object.
    """
    df = df.copy()

    # Prepare design matrix and dependent variable; coerce to float
    X_cols = ['StudentTeacherRatio', 'ExpenditurePerStudent', 'PctReducedLunch', 'PctCalWorks', 'PctEnglishLearners', 'Computers', 'Is_KK08']
    X = df[X_cols].astype(float)
    X = sm.add_constant(X)
    y = df['AvgTestScore'].astype(float)

    # Fit OLS with robust (HC3) standard errors
    model = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

    # Return the fitted model object (user can call .summary() on it)
    return model


