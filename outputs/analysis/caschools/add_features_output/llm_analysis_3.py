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
    # Work on a copy
    df = df.copy()

    # Ensure necessary columns exist and coerce numeric columns
    numeric_cols = ['students', 'teachers', 'read', 'math', 'calworks', 'lunch', 'income', 'english', 'expenditure', 'computer']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Required columns for the analytic sample
    required_cols = ['students', 'teachers', 'read', 'math', 'calworks', 'lunch', 'income', 'english', 'expenditure', 'computer', 'grades', 'county']

    # Drop rows missing any required column or with nonpositive teachers/students
    df = df.dropna(subset=required_cols)
    df = df[df['teachers'] > 0]
    df = df[df['students'] > 0]

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Controls: rename and compute derived controls
    df['PercentCalWorks'] = df['calworks']
    df['PercentReducedLunch'] = df['lunch']
    df['IncomeK'] = df['income']  # income already in thousands per schema
    df['PercentEnglishLearners'] = df['english']
    df['ExpenditurePerStudent'] = df['expenditure']

    # Computers per student
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log enrollment
    df['LogEnrollment'] = np.log(df['students'].astype(float))

    # Keep grade span and county as categorical columns with consistent names used in model
    df['Grades'] = df['grades'].astype(str).fillna('Unknown')
    df['County'] = df['county'].astype(str).fillna('Unknown')

    # Keep only columns needed for modeling (this also implicitly drops unwanted raw columns)
    model_cols = ['AvgScore', 'StudentTeacherRatio', 'PercentCalWorks', 'PercentReducedLunch', 'IncomeK',
                  'PercentEnglishLearners', 'ExpenditurePerStudent', 'ComputersPerStudent', 'LogEnrollment',
                  'Grades', 'County']
    df = df[model_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for socio-economic
    and resource variables plus categorical controls for Grades and County.

    Model specification (linear OLS):
      AvgScore_i = beta0 + beta1 * StudentTeacherRatio_i + gamma' * Controls_i + delta_Grades + theta_County + epsilon_i

    Returns the fitted statsmodels regression results object.
    """

    # Make a copy
    df = df.copy()

    # Prepare numeric covariates
    covariates = [
        'StudentTeacherRatio',
        'PercentCalWorks',
        'PercentReducedLunch',
        'IncomeK',
        'PercentEnglishLearners',
        'ExpenditurePerStudent',
        'ComputersPerStudent',
        'LogEnrollment'
    ]

    # Ensure no missing values remain in modeling columns
    modeling_cols = ['AvgScore'] + covariates + ['Grades', 'County']
    df = df.dropna(subset=modeling_cols)

    # Create categorical dummies for Grades and County (drop_first to avoid collinearity)
    grades_dummies = pd.get_dummies(df['Grades'], prefix='G', drop_first=True)
    county_dummies = pd.get_dummies(df['County'], prefix='C', drop_first=True)

    # Combine regressors
    X = df[covariates].reset_index(drop=True)
    X = pd.concat([X, grades_dummies.reset_index(drop=True), county_dummies.reset_index(drop=True)], axis=1)

    # Add constant
    X = sm.add_constant(X)
    y = df['AvgScore'].astype(float)

    # Fit OLS
    results = sm.OLS(y, X).fit()

    # Print a concise summary and return results
    print(results.summary())
    return results


