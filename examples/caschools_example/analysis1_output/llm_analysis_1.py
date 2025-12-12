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

    # Ensure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'computer']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the primary variables needed to compute the key measures
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Exclude rows with non-positive teacher counts to avoid division by zero
    df = df[df['teachers'] > 0]

    # Compute student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student (may be small); guard against division issues
    df['ComputersPerStudent'] = df['computer'] / df['students']
    df['ComputersPerStudent'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Size control: log of students (use log1p to handle small counts safely)
    df['LogStudents'] = np.log1p(df['students'])

    # Ensure categorical control is a category dtype
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')

    # Drop rows with NA in controls that will be used in the model to keep the same sample across regressors
    required_for_model = ['StudentTeacherRatio', 'AvgScore', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'LogStudents', 'grades']
    # Only include those required columns that are present in df
    required_present = [c for c in required_for_model if c in df.columns]
    df = df.dropna(subset=required_present)

    # Final returned dataframe contains original columns plus derived columns used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Define formula: effect of student-teacher ratio on average score controlling for resources and demographics
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english '
        '+ ComputersPerStudent + LogStudents + C(grades)'
    )

    # Fit OLS with robust (heteroskedasticity-consistent) standard errors
    model = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Return the fitted results object for inspection (summary, params, conf_int, etc.)
    return model


