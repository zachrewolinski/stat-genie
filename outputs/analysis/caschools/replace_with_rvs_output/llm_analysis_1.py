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
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'read', 'math']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing core variables needed for analysis
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove rows with non-positive teachers to avoid division by zero
    df = df[df['teachers'] > 0]

    # Compute student-teacher ratio (students per teacher)
    df['students_per_teacher'] = df['students'] / df['teachers']

    # Compute computers per student (handle division by zero defensively)
    df['computers_per_student'] = df['computer'] / df['students']

    # Compute the dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Create a binary indicator for grades span (KK-08 vs KK-06)
    # If grades is missing, set to 0 (will have been dropped earlier if core vars missing)
    df['grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)

    # Optional: winsorize extreme student-teacher ratios to reduce influence of outliers
    # Clip at 1st and 99th percentiles
    if 'students_per_teacher' in df.columns:
        lower = df['students_per_teacher'].quantile(0.01)
        upper = df['students_per_teacher'].quantile(0.99)
        df['students_per_teacher'] = df['students_per_teacher'].clip(lower=lower, upper=upper)

    # Ensure control columns exist (if missing, create as NaN so downstream code can handle/drop)
    for col in ['expenditure', 'income', 'lunch', 'english', 'calworks', 'students', 'computers_per_student']:
        if col not in df.columns:
            df[col] = np.nan

    # Final drop: remove any rows missing the dependent or the primary independent variable
    df = df.dropna(subset=['AvgScore', 'students_per_teacher'])

    # Return the transformed dataframe containing all columns used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Copy to avoid modifying original
    df = df.copy()

    # Columns to include in the regression
    X_cols = [
        'students_per_teacher',
        'expenditure',
        'income',
        'lunch',
        'english',
        'computers_per_student',
        'calworks',
        'grades_KK08',
        'students'
    ]

    # Keep only rows with no missing values on model predictors (or alternatively drop/ impute as needed)
    df_model = df.dropna(subset=X_cols + ['AvgScore'])

    # Design matrix
    X = df_model[X_cols].astype(float)
    X = sm.add_constant(X)
    y = df_model['AvgScore'].astype(float)

    # Fit OLS with heteroskedasticity-robust standard errors (HC3)
    model_res = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted results object. The caller can inspect model_res.summary(), model_res.params, etc.
    return model_res


