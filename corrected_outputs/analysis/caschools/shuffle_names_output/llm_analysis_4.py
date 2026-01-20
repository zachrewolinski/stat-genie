from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/shuffle_names_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Map and coerce numeric columns (dataset column names used below):
    # - 'calworks' appears to contain total enrollment (students)
    # - 'teachers' contains number of teachers (FTE)
    # - 'expenditure' is expenditure per student
    # - 'income' is district income / poverty indicator
    # - 'english' in this file appears to represent number of computers per the provided schema; call it 'Computers'
    # - 'grades' and 'rownames' contain standardized test scores (two score variables). We'll average them for the outcome.

    df['Students'] = pd.to_numeric(df.get('calworks'), errors='coerce')
    df['Teachers'] = pd.to_numeric(df.get('teachers'), errors='coerce')
    df['Expenditure'] = pd.to_numeric(df.get('expenditure'), errors='coerce')
    df['Income'] = pd.to_numeric(df.get('income'), errors='coerce')
    df['Computers'] = pd.to_numeric(df.get('english'), errors='coerce')

    # Scores
    df['GradesScore'] = pd.to_numeric(df.get('grades'), errors='coerce')
    df['MathScore'] = pd.to_numeric(df.get('rownames'), errors='coerce')

    # Outcome: average of available test score columns (will be NaN if both missing)
    df['AvgScore'] = df[['GradesScore', 'MathScore']].mean(axis=1)

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['Students'] / df['Teachers']

    # Basic cleaning: drop rows with missing key model variables
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgScore'])

    # Keep only rows with finite, positive ratios
    df = df[df['StudentTeacherRatio'].replace([np.inf, -np.inf], np.nan).notnull()]
    df = df[df['StudentTeacherRatio'] > 0]

    # School type (categorical control)
    df['SchoolType'] = df['school'].astype(str)

    # Final returned dataframe should contain all columns used in the model
    final_cols = [
        'Students',
        'Teachers',
        'StudentTeacherRatio',
        'AvgScore',
        'Expenditure',
        'Income',
        'Computers',
        'SchoolType'
    ]

    # Some control variables may be partially missing; keep rows and let model function handle any remaining missingness
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Work on a copy
    df = df.copy()

    # Prepare independent variables matrix X and dependent variable y
    # Use the transformed columns exactly as defined in the transform function
    X = df[['StudentTeacherRatio', 'Expenditure', 'Income', 'Computers']].copy()

    # For any remaining NA in control variables, impute with column mean (simple, transparent choice).
    # Alternatively one could drop rows with missing controls; here we impute to preserve sample size.
    X = X.fillna(X.mean())

    # Add categorical dummies for SchoolType (drop first to avoid multicollinearity)
    school_dummies = pd.get_dummies(df['SchoolType'].astype(str), prefix='SchoolType', drop_first=True)
    if not school_dummies.empty:
        X = pd.concat([X, school_dummies], axis=1)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Dependent variable
    y = df['AvgScore']

    # Fit OLS with robust (HC3) standard errors to guard against heteroskedasticity
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HC3')

    # Return the fitted results object (contains params, summary(), etc.)
    return results


