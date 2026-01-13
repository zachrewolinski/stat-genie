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
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following derived columns (required by the model):
    - student_teacher_ratio: students / teachers
    - AvgScore: mean of 'read' and 'math'
    - computer_per_student: computer / students

    Also ensures categorical columns have categorical dtype and drops rows with critical missing
    values or invalid teacher/student counts.
    """
    df = df.copy()

    # Relevant columns we will use/derive
    needed_cols = [
        'students', 'teachers', 'read', 'math',
        'expenditure', 'income', 'lunch', 'calworks', 'english', 'computer',
        'grades', 'county'
    ]

    # If any of these columns do not exist in the dataframe, raise a clear error
    missing = [c for c in needed_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing essential values for computing the DV/IV
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove invalid teacher/student values (teachers <= 0 or students <= 0)
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Compute student-teacher ratio (students per teacher)
    df['student_teacher_ratio'] = df['students'] / df['teachers']

    # Compute average score (dependent variable)
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Compute computers per student as a control; allow float and handle zeros already filtered
    df['computer_per_student'] = df['computer'] / df['students']

    # Convert categorical controls to category dtype for modeling with dummy variables
    df['grades'] = df['grades'].astype('category')
    df['county'] = df['county'].astype('category')

    # Keep original read/math for traceability and also keep columns used in modeling
    # (no further scaling here; modeling code can standardize if desired)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs an OLS regression to estimate the association between student-teacher ratio
    and average academic performance, controlling for district resources and
    demographics. Uses robust (HC3) standard errors.

    Formula:
      AvgScore ~ student_teacher_ratio + expenditure + income + lunch + calworks + english + computer_per_student + C(grades) + C(county)

    Returns:
      statsmodels RegressionResults object (fitted model).
    """
    import statsmodels.formula.api as smf

    # Ensure required modeling columns exist
    model_cols = [
        'AvgScore', 'student_teacher_ratio', 'expenditure', 'income', 'lunch',
        'calworks', 'english', 'computer_per_student', 'grades', 'county'
    ]
    missing = [c for c in model_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Dataframe missing columns required for modeling: {missing}")

    # Drop rows with missing values in modeling columns
    df_model = df.dropna(subset=model_cols).copy()

    # Define formula. County and grades are treated as categorical fixed effects.
    formula = (
        'AvgScore ~ student_teacher_ratio + expenditure + income + lunch + '
        'calworks + english + computer_per_student + C(grades) + C(county)'
    )

    # Fit OLS with robust standard errors (HC3 to help with small-sample heteroskedasticity)
    results = smf.ols(formula=formula, data=df_model).fit(cov_type='HC3')

    return results


