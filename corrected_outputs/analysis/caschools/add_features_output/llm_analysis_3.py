from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district dataset to produce the columns needed for modeling:
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of read and math
      - Expenditure, Income, PctLunch, PctEnglishLearners, PctCalWorks (renamed for clarity)
      - ComputersPerStudent: computer / students
      - LogStudents: ln(students)
      - Grades_KK08: binary indicator for grades == 'KK-08'

    The function drops rows with missing or invalid core values (students, teachers, read, math)
    and avoids division by zero.
    """

    df = df.copy()

    # Keep original numeric columns that we will use; ensure they exist
    required_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'lunch', 'english', 'calworks', 'computer', 'grades']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing core values needed to compute IV and DV
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove invalid (<= 0) values for students or teachers to avoid division by zero or nonsensical ratios
    df = df[(df['students'] > 0) & (df['teachers'] > 0)]

    # Compute student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Rename/control variables for clarity
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']
    df['PctLunch'] = df['lunch']
    df['PctEnglishLearners'] = df['english']
    df['PctCalWorks'] = df['calworks']

    # Computers per student (avoid division by zero because students > 0 by earlier filter)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log of total enrollment: use natural log
    # add a small positive constant if extremely small values were present, but we've already dropped students <= 0
    df['LogStudents'] = np.log(df['students'])

    # Binary indicator for grade span: 1 if 'KK-08', 0 otherwise (including 'KK-06')
    # Ensure grades are strings before comparison
    df['Grades_KK08'] = df['grades'].astype(str).apply(lambda x: 1 if x == 'KK-08' else 0)

    # Keep only the final columns needed for modeling to make downstream code simpler
    final_cols = [
        'StudentTeacherRatio', 'AvgScore', 'Expenditure', 'Income', 'PctLunch',
        'PctEnglishLearners', 'PctCalWorks', 'ComputersPerStudent', 'LogStudents', 'Grades_KK08'
    ]

    # Some control columns may be fully missing; keep them as-is (modeling will drop rows with NA), but ensure final dataframe contains these column names
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[final_cols]

    # Drop rows with NA in the dependent variable or the independent variable (StudentTeacherRatio)
    df = df.dropna(subset=['AvgScore', 'StudentTeacherRatio'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for district-level covariates.
    Uses heteroskedasticity-robust (HC3) standard errors.

    Model specification:
      AvgScore ~ StudentTeacherRatio + Expenditure + Income + PctLunch + PctEnglishLearners
                 + PctCalWorks + ComputersPerStudent + LogStudents + Grades_KK08

    Returns the fitted statsmodels RegressionResults instance (with robust cov).
    """

    # Work on a copy
    df_model = df.copy()

    # Drop rows with missing values in any regressors or the outcome (complete-case analysis)
    regressors = [
        'StudentTeacherRatio', 'Expenditure', 'Income', 'PctLunch', 'PctEnglishLearners',
        'PctCalWorks', 'ComputersPerStudent', 'LogStudents', 'Grades_KK08'
    ]
    required = ['AvgScore'] + regressors
    df_model = df_model.dropna(subset=required)

    # Design matrix X and outcome y
    X = df_model[regressors]
    X = sm.add_constant(X)
    y = df_model['AvgScore']

    # Fit OLS with robust standard errors (HC3)
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC3')

    # For convenience, attach the dataframe used for the regression to the results object (optional)
    try:
        results.model_data = df_model
    except Exception:
        pass

    return results


