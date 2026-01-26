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
    Transform the raw district dataframe to produce the columns needed for modeling.

    Produces:
      - AvgScore: mean of 'read' and 'math'
      - StudentTeacherRatio: students / teachers
      - ComputersPerStudent: computer / students
      - LogStudents: log1p(students)

    It drops rows with missing values in any columns required for the model.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the core outcome or core size/teacher values
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Dependent variable: average test score (read & math)
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio
    # Protect against division by zero
    df['StudentTeacherRatio'] = df['students'] / df['teachers']
    df.loc[~np.isfinite(df['StudentTeacherRatio']), 'StudentTeacherRatio'] = np.nan

    # Resource control: computers per student
    df['ComputersPerStudent'] = df['computer'] / df['students']
    df.loc[~np.isfinite(df['ComputersPerStudent']), 'ComputersPerStudent'] = np.nan

    # District size (log transform)
    df['LogStudents'] = np.log1p(df['students'])

    # Ensure categorical fields exist and are of type category
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Keep only the columns needed for modeling (plus a few originals for traceability)
    needed_cols = [
        'AvgScore', 'StudentTeacherRatio', 'income', 'calworks', 'lunch', 'english',
        'expenditure', 'ComputersPerStudent', 'LogStudents', 'grades', 'county',
        'students', 'teachers', 'computer', 'read', 'math'
    ]
    # Select intersection of needed_cols and actual columns to avoid KeyError
    cols_to_keep = [c for c in needed_cols if c in df.columns]
    df = df[cols_to_keep]

    # Final drop: remove rows with any missing values among the columns that will be used in the model
    df = df.dropna(subset=[c for c in ['AvgScore', 'StudentTeacherRatio', 'income', 'calworks', 'lunch', 'english', 'expenditure', 'ComputersPerStudent', 'LogStudents', 'grades', 'county'] if c in df.columns])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of average test score on student-teacher ratio with controls.

    Returns the fitted statsmodels regression results object (with robust standard errors).
    """
    import statsmodels.formula.api as smf

    # Build the formula. Include categorical controls for grades and county if present.
    formula_parts = [
        'StudentTeacherRatio',
        'income',
        'calworks',
        'lunch',
        'english',
        'expenditure',
        'ComputersPerStudent',
        'LogStudents'
    ]

    # Start formula
    formula = 'AvgScore ~ ' + ' + '.join(formula_parts)

    # Add categorical terms if present in df
    if 'grades' in df.columns:
        formula += ' + C(grades)'
    if 'county' in df.columns:
        formula += ' + C(county)'

    # Fit OLS with robust (HC3) standard errors
    model = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object. The caller can inspect model.summary() or model.params etc.
    return model


