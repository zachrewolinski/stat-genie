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
    Prepare the dataset for modeling. The transformations performed:
      - Drop rows missing the key variables needed to compute the DV or IV (students, teachers, read, math).
      - Remove rows with nonpositive teacher counts (to avoid division by zero).
      - Create AvgScore = (read + math) / 2.
      - Create StudentTeacherRatio = students / teachers (students per teacher).
      - Create ComputerPerStudent = computer / students (handle zero enrollment safely by setting NA if students==0).
      - Create LogStudents = np.log1p(students) to reduce skew.
      - Ensure grades and county are categorical.
    The returned dataframe contains all columns used in the model.
    """
    # Make a copy to avoid modifying input in-place
    df = df.copy()

    # Drop rows missing the required variables for DV and IV
    required_cols = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=required_cols)

    # Remove rows where teachers is zero or negative to prevent division by zero
    df = df[df['teachers'] > 0]

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Computer resources: computers per student (set to NaN if students == 0)
    df['ComputerPerStudent'] = np.where(df['students'] > 0, df['computer'] / df['students'], np.nan)

    # Log enrollment to control for size (use log1p to handle small values)
    df['LogStudents'] = np.log1p(df['students'].astype(float))

    # Ensure categorical controls are typed properly
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Keep only columns needed for modeling plus identifiers for traceability
    model_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'ComputerPerStudent',
        'LogStudents',
        # socioeconomic / resource controls present in the original dataset
        'expenditure',
        'income',
        'lunch',
        'english',
        'grades',
        'county'
    ]

    # Some datasets may not contain all control columns; ensure they exist in the returned df (fill missing with NaN)
    for c in model_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Return dataframe limited to the model columns plus any index/identifier columns (if present)
    # Keep original index
    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and academic performance.

    Model specification (controls included):
      AvgScore ~ StudentTeacherRatio + expenditure + income + lunch + english
                 + ComputerPerStudent + LogStudents + C(grades) + C(county)

    County and grades are included as categorical fixed effects. Robust (HC3) standard errors are used.

    The function returns the fitted results object (statsmodels RegressionResults).
    """
    import statsmodels.formula.api as smf

    # Ensure the columns used in the formula exist in df (precondition: df is output of transform)
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + lunch + english '
        '+ ComputerPerStudent + LogStudents + C(grades) + C(county)'
    )

    # Fit model using OLS with robust standard errors (HC3)
    # Drop rows with NA in the variables used by the model
    vars_in_formula = [
        'AvgScore', 'StudentTeacherRatio', 'expenditure', 'income', 'lunch', 'english',
        'ComputerPerStudent', 'LogStudents', 'grades', 'county'
    ]
    df_model = df.dropna(subset=['AvgScore', 'StudentTeacherRatio'])
    # It's ok if some controls are NA; statsmodels will drop rows automatically, but explicitly drop rows
    df_model = df_model.dropna(subset=['expenditure', 'income', 'lunch', 'english', 'ComputerPerStudent', 'LogStudents'], how='any')

    results = smf.ols(formula=formula, data=df_model).fit(cov_type='HC3')

    # Return the fitted results object so the caller can inspect .summary(), params, conf_int(), etc.
    return results


