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
    Transform the raw dataset into a dataframe ready for modeling. Adds derived variables used in the model.

    Columns created/returned (in addition to original columns):
      - stu_teacher_ratio: students / teachers (FTE). Rows with teachers <= 0 are removed.
      - AvgScore: mean of 'read' and 'math'.
      - computer_per_student: computer / students (set to NaN when students <= 0).
      - grades_KK08: binary indicator 1 if grades == 'KK-08', 0 otherwise (including 'KK-06').
      - log_students: natural log of students.

    Rows with missing values in key columns (students, teachers, read, math) will be dropped.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    num_cols = ['students', 'teachers', 'computer', 'read', 'math', 'expenditure', 'income', 'english', 'lunch', 'calworks']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing critical values
    required = ['students', 'teachers', 'read', 'math']
    present_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=present_required)

    # Remove rows where teachers is zero or negative (invalid for ratio)
    df = df[df['teachers'] > 0]

    # Derive student-teacher ratio
    df['stu_teacher_ratio'] = df['students'] / df['teachers']

    # Dependent variable: average of read and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student (resource control); guard division by zero
    df['computer_per_student'] = df['computer'] / df['students']

    # Grade-span indicator: KK-08 vs KK-06 (handle NaN gracefully)
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype(str)
        df['grades_KK08'] = (df['grades'] == 'KK-08').astype(int)
    else:
        df['grades_KK08'] = 0

    # Log of students to control for scale
    df['log_students'] = np.log(df['students'].replace({0: np.nan}))

    # Keep all original columns plus the derived ones; explicit return
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs OLS regressions to estimate association between student-teacher ratio and academic performance.

    Returns a dictionary with two fitted models:
      - 'unadjusted': AvgScore ~ stu_teacher_ratio
      - 'adjusted': AvgScore ~ stu_teacher_ratio + controls

    Both models use robust (HC3) standard errors.
    """
    # Make a local copy
    data = df.copy()

    # Ensure the transform has been run; check for required columns
    required_cols = ['AvgScore', 'stu_teacher_ratio']
    for c in required_cols:
        if c not in data.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe. Please run transform() first.")

    # Unadjusted model
    X_unadj = sm.add_constant(data[['stu_teacher_ratio']], has_constant='add')
    y_unadj = data['AvgScore']
    # Drop rows with missing values in the used columns
    mask_unadj = X_unadj.notnull().all(axis=1) & y_unadj.notnull()
    X_unadj_clean = X_unadj[mask_unadj]
    y_unadj_clean = y_unadj[mask_unadj]
    model_unadj = sm.OLS(y_unadj_clean, X_unadj_clean).fit(cov_type='HC3')

    # Adjusted model: add controls defined in transform
    controls = [
        'expenditure', 'income', 'english', 'lunch', 'calworks',
        'computer_per_student', 'grades_KK08', 'log_students'
    ]
    present_controls = [c for c in controls if c in data.columns]

    model_vars = ['stu_teacher_ratio'] + present_controls
    # Drop rows with missingness in any model variable or outcome
    df_model = data.dropna(subset=model_vars + ['AvgScore'])

    X_full = sm.add_constant(df_model[model_vars], has_constant='add')
    y_full = df_model['AvgScore']

    model_full = sm.OLS(y_full, X_full).fit(cov_type='HC3')

    # Return both fitted model objects and the data used for the adjusted model for reproducibility
    return {
        'unadjusted': model_unadj,
        'adjusted': model_full,
        'model_data_adjusted': df_model  # useful for diagnostics / further checks
    }

# Example usage:
# df_transformed = transform(raw_df)
# results = model(df_transformed)
# print(results['adjusted'].summary())


