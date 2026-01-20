from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/noperturb_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations performed:
    - Drop rows with missing essential variables (students, teachers, read, math).
    - Remove rows with nonpositive teachers (to avoid division by zero).
    - Compute StudentTeacherRatio = students / teachers.
    - Compute AvgScore = (read + math) / 2.
    - Compute ComputersPerStudent = computer / students.
    - Compute LogStudents = log(students).
    - Create grades_KK08 indicator column (1 if grades == 'KK-08', else 0).
    - Create county dummy variables with prefix 'county_' (drop first to avoid multicollinearity).

    Returns dataframe augmented with all columns used by the model.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing essential data
    required = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=required)

    # Remove rows with nonpositive teachers to avoid division by zero
    df = df[df['teachers'] > 0]

    # Dependent variable: average of reading and math
    df['AvgScore'] = (df['read'] + df['math']) / 2.0

    # Independent variable: students per teacher (student-teacher ratio)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Controls derived
    # Computers per student (handle division safely)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log of students (size)
    df['LogStudents'] = np.log(df['students'].replace(0, np.nan))

    # Grade span indicator: 1 if KK-08 else 0 (KK-06 -> 0). If missing, fill 0.
    df['grades_KK08'] = (df['grades'].fillna('KK-06') == 'KK-08').astype(int)

    # County fixed effects: get dummies, drop first to be reference
    if 'county' in df.columns:
        county_dummies = pd.get_dummies(df['county'].astype(str), prefix='county', drop_first=True)
        # Attach county dummy columns to df
        df = pd.concat([df.reset_index(drop=True), county_dummies.reset_index(drop=True)], axis=1)

    # Keep only columns needed for modeling (plus original ids/info if desired)
    # List of columns that will be used downstream
    model_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'expenditure',
        'income',
        'calworks',
        'lunch',
        'english',
        'ComputersPerStudent',
        'LogStudents',
        'grades_KK08'
    ]
    # Add any county dummy columns
    county_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('county_')]
    model_cols += county_cols

    # Return dataframe filtered to rows that have no missing values in model columns
    df = df.dropna(subset=model_cols)

    # Ensure final df contains model columns (and leave other columns as well)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for multiple district covariates and county fixed effects.
    Uses heteroskedasticity-robust standard errors (HC3).

    Returns the fitted regression results (statsmodels RegressionResults wrapper).
    """
    # Ensure the transformed dataframe has the columns we need
    required_cols = ['AvgScore', 'StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'LogStudents', 'grades_KK08']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    y = df['AvgScore']

    # Base covariates
    X_cols = ['StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'LogStudents', 'grades_KK08']

    # Add county fixed effects if present (columns starting with 'county_')
    county_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('county_')]
    X_cols += county_cols

    X = df[X_cols]
    X = sm.add_constant(X)

    # Fit OLS with robust standard errors (HC3)
    model_res = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted results object (user can call .summary() on it)
    return model_res


