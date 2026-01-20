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
    # Make a copy to avoid modifying original
    df = df.copy()

    # Convert columns that should be numeric to numeric, coercing errors to NaN
    numeric_cols = ['calworks', 'teachers', 'grades', 'rownames', 'expenditure', 'income', 'math', 'district', 'english']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Rename / create clear column names for downstream modeling
    # Total enrollment: use 'calworks' per dataset description
    df['TotalEnrollment'] = df.get('calworks')
    # Number of teachers (FTE)
    df['NumTeachers'] = df.get('teachers')

    # Compute the student-teacher ratio where possible
    # Avoid division by zero and invalid values
    df['StudentTeacherRatio'] = np.where(
        (df['NumTeachers'].notna()) & (df['NumTeachers'] > 0) & (df['TotalEnrollment'].notna()),
        df['TotalEnrollment'] / df['NumTeachers'],
        np.nan
    )

    # Logged ratio to reduce skew; only for positive ratios
    df['LogStudentTeacherRatio'] = np.where(df['StudentTeacherRatio'] > 0,
                                            np.log(df['StudentTeacherRatio']),
                                            np.nan)

    # Construct the dependent variable: average of reading and math scores where available
    # 'grades' is described as average reading score; 'rownames' as average math score
    if ('grades' in df.columns) and ('rownames' in df.columns):
        df['AvgTestScore'] = df[['grades', 'rownames']].mean(axis=1)
    elif 'grades' in df.columns:
        df['AvgTestScore'] = df['grades']
    elif 'rownames' in df.columns:
        df['AvgTestScore'] = df['rownames']
    else:
        df['AvgTestScore'] = np.nan

    # Controls: map dataset columns to clear control variable names
    df['ExpenditurePerStudent'] = df.get('expenditure')  # per-dataset: expenditure per student
    df['PctCalWorks'] = df.get('income')                 # percent qualifying for CalWorks
    df['PctReducedLunch'] = df.get('math')               # percent qualifying for reduced-price lunch
    df['PctEnglishLearners'] = df.get('district')        # percent English learners
    df['ComputersPerClassroom'] = df.get('english')      # raw number of computers (mapped directly)

    # Drop rows missing the dependent variable or the primary independent variable
    # Model will use the logged ratio; require both AvgTestScore and LogStudentTeacherRatio
    df = df.dropna(subset=['AvgTestScore', 'LogStudentTeacherRatio'])

    # Optionally, remove observations with implausible values (basic winsorization)
    # Remove extremely large ratios (e.g., > 200 students per teacher) as likely data errors
    df.loc[df['StudentTeacherRatio'] > 200, ['StudentTeacherRatio', 'LogStudentTeacherRatio']] = np.nan
    df = df.dropna(subset=['AvgTestScore', 'LogStudentTeacherRatio'])

    # Keep only columns needed for modeling plus helpful diagnostics
    keep_cols = [
        'TotalEnrollment', 'NumTeachers', 'StudentTeacherRatio', 'LogStudentTeacherRatio',
        'AvgTestScore', 'ExpenditurePerStudent', 'PctCalWorks', 'PctReducedLunch',
        'PctEnglishLearners', 'ComputersPerClassroom'
    ]
    # Add other original columns if they exist for potential diagnostics
    for c in ['county', 'lunch', 'students', 'school', 'computer']:
        if c in df.columns:
            keep_cols.append(c)

    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Ensure a copy is used
    df = df.copy()

    # Select independent variable and controls
    iv = 'LogStudentTeacherRatio'
    dv = 'AvgTestScore'
    controls = [
        'ExpenditurePerStudent',
        'PctCalWorks',
        'PctReducedLunch',
        'PctEnglishLearners',
        'ComputersPerClassroom'
    ]

    # Keep only rows with no missing values in the model variables
    model_vars = [iv, dv] + controls
    model_df = df.dropna(subset=model_vars)

    # Prepare design matrices
    X = model_df[[iv] + controls]
    X = sm.add_constant(X)
    y = model_df[dv]

    # Fit OLS and compute robust standard errors (HC3)
    ols_res = sm.OLS(y, X).fit()
    res_robust = ols_res.get_robustcov_results(cov_type='HC3')

    # Return the robust-results object (has summary(), params, bse, pvalues, etc.)
    return res_robust


