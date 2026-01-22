from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/replace_and_positive_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform original dataset to include the variables used in the analysis.

    Creates:
    - StudentTeacherRatio: students / teachers (winsorized at 1st/99th pct)
    - AvgScore: mean of read and math
    - LogStudents: natural log of students
    - County: categorical copy of the county variable

    Keeps columns needed for the model and drops rows with missing critical values.
    """
    df = df.copy()

    # Ensure numeric columns are numeric where possible
    numeric_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'english', 'lunch', 'calworks']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing key inputs
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Compute student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Winsorize extreme ratio values at 1st and 99th percentiles to reduce influence of outliers
    lower = df['StudentTeacherRatio'].quantile(0.01)
    upper = df['StudentTeacherRatio'].quantile(0.99)
    df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower, upper)

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Control: log of students (district size)
    # add a small constant guard (not necessary given min students > 0 in schema) but keep for safety
    df['LogStudents'] = np.log(df['students'].clip(lower=1))

    # County as categorical for fixed effects
    if 'county' in df.columns:
        df['County'] = df['county'].astype('category')
    else:
        df['County'] = pd.Series(['Unknown'] * len(df), index=df.index).astype('category')

    # Keep the columns that the model will use (plus read/math for robustness checks)
    keep_cols = ['StudentTeacherRatio', 'AvgScore', 'LogStudents', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'County', 'read', 'math']
    # Keep only columns present in df
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits OLS models to estimate the association between student-teacher ratio and academic performance.

    Main specification:
      AvgScore ~ StudentTeacherRatio + LogStudents + expenditure + income + english + lunch + calworks + C(County)

    Uses heteroskedasticity-robust (HC3) standard errors. Also fits robustness checks on the reading and math outcomes separately.

    Returns a dict with fitted statsmodels results objects: {'main': ..., 'read': ..., 'math': ...}
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Drop rows with missing values in model variables
    required = ['StudentTeacherRatio', 'AvgScore', 'LogStudents', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'County']
    required_present = [c for c in required if c in df.columns]
    df = df.dropna(subset=required_present)

    # Main model formula (AvgScore)
    formula_main = 'AvgScore ~ StudentTeacherRatio + LogStudents + expenditure + income + english + lunch + calworks + C(County)'
    res_main = smf.ols(formula_main, data=df).fit(cov_type='HC3')

    # Robustness: separate models for reading and math
    results = {'main': res_main}
    if 'read' in df.columns:
        formula_read = 'read ~ StudentTeacherRatio + LogStudents + expenditure + income + english + lunch + calworks + C(County)'
        results['read'] = smf.ols(formula_read, data=df).fit(cov_type='HC3')
    if 'math' in df.columns:
        formula_math = 'math ~ StudentTeacherRatio + LogStudents + expenditure + income + english + lunch + calworks + C(County)'
        results['math'] = smf.ols(formula_math, data=df).fit(cov_type='HC3')

    return results


