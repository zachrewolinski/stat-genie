from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/negative_leading_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe to produce the analysis-ready variables.

    Produces the following new columns used in modeling:
      - StudentTeacherRatio: students / teachers
      - AvgScore: average of read and math scores
      - ComputerPerStudent: computer / students
      - log_students: natural log of students
      - grades_KK08: binary indicator (1 if grades == 'KK-08')

    The function drops rows with missing values for core quantities (students, teachers, read, math).
    """
    df = df.copy()

    # Ensure numeric columns exist and handle obvious invalid values
    # Replace zero or negative teachers with NaN to avoid division by zero
    if 'teachers' in df.columns:
        df.loc[df['teachers'] <= 0, 'teachers'] = np.nan

    # Drop rows missing the core variables required to compute the primary IV and DV
    required = ['students', 'teachers', 'read', 'math']
    present_required = [c for c in required if c in df.columns]
    if len(present_required) < len(required):
        missing = set(required) - set(present_required)
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=present_required)

    # Create student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student (handle zero students defensively)
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Log of students to control for district size (avoid log(0) by filtering above)
    df['log_students'] = np.log(df['students'])

    # Binary indicator for grade span KK-08 (1 if KK-08, else 0)
    # Ensure grades column exists
    if 'grades' in df.columns:
        df['grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)
    else:
        df['grades_KK08'] = 0

    # Keep the county column as-is for fixed effects in the model; ensure it's string/categorical
    if 'county' in df.columns:
        df['county'] = df['county'].astype(str)

    # It's useful to drop rows that still have NA in any of the model columns
    model_vars = ['StudentTeacherRatio', 'AvgScore', 'expenditure', 'income', 'ComputerPerStudent', 'calworks', 'lunch', 'english', 'log_students', 'grades_KK08', 'county']
    # Only require those model_vars that are present in df
    model_vars_present = [v for v in model_vars if v in df.columns]
    df = df.dropna(subset=model_vars_present)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit an OLS model testing the association between StudentTeacherRatio and AvgScore,
    controlling for district socioeconomic and resource covariates and county fixed effects.

    The main coefficient of interest is on StudentTeacherRatio. We use robust (HC3)
    standard errors to protect against heteroskedasticity.

    Returns the robust results object.
    """
    import statsmodels.formula.api as smf

    # Make a working copy
    df = df.copy()

    # Ensure required modeling columns exist
    required = ['AvgScore', 'StudentTeacherRatio']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Formula includes controls; C(county) will create county fixed effects
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + ComputerPerStudent '
        '+ calworks + lunch + english + log_students + grades_KK08 + C(county)'
    )

    # Fit OLS and compute robust covariance (HC3)
    ols_fit = smf.ols(formula=formula, data=df).fit()
    robust_results = ols_fit.get_robustcov_results(cov_type='HC3')

    # Print a concise summary to assist immediate interpretation
    print(robust_results.summary())

    # Return the robust results object for further inspection
    return robust_results


