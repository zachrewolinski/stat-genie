from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district dataset into a cleaned dataframe ready for modeling.
    Adds the following columns required by the model:
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of 'read' and 'math'
      - Expenditure, LunchPct, EnglishLearnersPct, Income, Students, LogStudents
      - County, Grades as categorical columns

    Drops rows with missing critical values (students, teachers, read, math).
    """
    df = df.copy()

    # Standardize column names for safety (work with given schema names)
    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'lunch', 'english', 'income']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core variables needed to compute ratio and outcome
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove rows with non-positive teachers to avoid division by zero
    df = df[df['teachers'] > 0]

    # Compute Student-Teacher Ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of read and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Map and copy control variables to the exact column names used in the model
    # Expenditure per student
    if 'expenditure' in df.columns:
        df['Expenditure'] = df['expenditure'].astype(float)
    else:
        df['Expenditure'] = np.nan

    # Percent qualifying for reduced-price lunch
    if 'lunch' in df.columns:
        df['LunchPct'] = df['lunch'].astype(float)
    else:
        df['LunchPct'] = np.nan

    # Percent English learners
    if 'english' in df.columns:
        df['EnglishLearnersPct'] = df['english'].astype(float)
    else:
        df['EnglishLearnersPct'] = np.nan

    # Income (district average income in 1,000s)
    if 'income' in df.columns:
        df['Income'] = df['income'].astype(float)
    else:
        df['Income'] = np.nan

    # Students (raw enrollment)
    df['Students'] = df['students'].astype(float)

    # Log of students (handle zeros by replacing with small positive value if any)
    df['LogStudents'] = np.log(df['Students'].replace(0, np.nan))

    # Categorical controls: County and Grades
    if 'county' in df.columns:
        df['County'] = df['county'].astype('category')
    else:
        df['County'] = pd.Categorical([np.nan] * len(df))

    if 'grades' in df.columns:
        df['Grades'] = df['grades'].astype('category')
    else:
        df['Grades'] = pd.Categorical([np.nan] * len(df))

    # Keep only columns necessary for modeling plus a few helpful raw columns
    keep_cols = [
        'StudentTeacherRatio', 'AvgScore', 'Expenditure', 'LunchPct',
        'EnglishLearnersPct', 'Income', 'Students', 'LogStudents',
        'County', 'Grades'
    ]

    # Ensure all keep_cols exist in dataframe (they should, but fill missing with NaN)
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio with controls.

    Model specification:
      AvgScore ~ StudentTeacherRatio + Expenditure + LunchPct + EnglishLearnersPct + Income + LogStudents + C(County)

    Uses robust (HC1) standard errors. Returns the fitted statsmodels regression results object.
    """
    import statsmodels.formula.api as smf

    # Drop rows with missing values in the variables used in the model
    required_vars = [
        'AvgScore', 'StudentTeacherRatio', 'Expenditure', 'LunchPct',
        'EnglishLearnersPct', 'Income', 'LogStudents', 'County'
    ]
    df_model = df.dropna(subset=required_vars).copy()

    # If LogStudents is still missing (e.g., Students was 0), try to compute from Students
    if 'LogStudents' in df_model.columns and df_model['LogStudents'].isna().any():
        df_model.loc[df_model['LogStudents'].isna(), 'LogStudents'] = np.log(df_model.loc[df_model['LogStudents'].isna(), 'Students'].replace(0, np.nan))
        df_model = df_model.dropna(subset=['LogStudents'])

    # Define formula including county fixed effects
    formula = 'AvgScore ~ StudentTeacherRatio + Expenditure + LunchPct + EnglishLearnersPct + Income + LogStudents + C(County)'

    # Fit OLS with robust standard errors (HC1)
    model_res = smf.ols(formula=formula, data=df_model).fit(cov_type='HC1')

    return model_res


