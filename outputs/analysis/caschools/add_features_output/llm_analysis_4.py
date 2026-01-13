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
    Transforms the input dataframe to include the variables required for modeling.

    - Drops observations missing key variables.
    - Removes observations with non-positive teacher counts.
    - Constructs StudentTeacherRatio, AvgScore, ComputersPerStudent, LogEnrollment,
      Grades_KK08 and ensures county is a string for later dummy creation.

    Returns the transformed dataframe containing the new columns used in the model.
    """
    df = df.copy()

    # Columns required to compute primary variables and controls
    required_cols = [
        'students', 'teachers', 'read', 'math',
        'expenditure', 'lunch', 'english', 'income', 'computer', 'grades', 'county'
    ]

    # Drop rows missing any of the required raw inputs
    df = df.dropna(subset=required_cols)

    # Remove invalid teacher counts (avoid division by zero)
    df = df[df['teachers'] > 0]

    # Student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student
    # if students == 0 (unlikely) this will produce inf/NaN; handled by subsequent dropna
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log enrollment (protect against zeros, but zeros were removed earlier by required_cols and students>0)
    df['LogEnrollment'] = np.log(df['students'].astype(float))

    # Binary indicator for grade span KK-08 (1) vs KK-06 (0)
    # Ensure comparison handles possible whitespace or casing issues
    df['Grades_KK08'] = (df['grades'].astype(str).str.strip() == 'KK-08').astype(int)

    # Ensure county is a string (categorical) for dummy creation in model
    df['county'] = df['county'].astype(str)

    # Final drop of any rows that produced NaNs in derived columns
    derived_cols = ['StudentTeacherRatio', 'AvgScore', 'ComputersPerStudent', 'LogEnrollment', 'Grades_KK08', 'county']
    df = df.dropna(subset=derived_cols)

    # Keep only columns that are needed (original controls kept in original names)
    keep_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ComputersPerStudent', 'LogEnrollment', 'Grades_KK08', 'county',
        'expenditure', 'lunch', 'english', 'income', 'students', 'teachers', 'read', 'math', 'computer', 'grades'
    ]

    # Some of these may not exist if original dataset used different casing; assume schema-provided names are correct
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fits an OLS regression of AvgScore on StudentTeacherRatio controlling for
    expenditure, lunch, english, income, ComputersPerStudent, LogEnrollment, Grades_KK08,
    and county fixed effects (implemented via dummies).

    Uses heteroskedasticity-robust (HC3) standard errors.
    Returns the fitted statsmodels results object.
    """
    df = df.copy()

    # Define baseline covariates (must match transformed dataframe column names)
    covariates = [
        'StudentTeacherRatio',
        'expenditure',
        'lunch',
        'english',
        'income',
        'ComputersPerStudent',
        'LogEnrollment',
        'Grades_KK08'
    ]

    # Verify covariates present
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required covariate columns in transformed df: {missing}")

    X = df[covariates].copy()

    # Create county dummies (drop first to avoid multicollinearity)
    county_dummies = pd.get_dummies(df['county'].astype(str), prefix='county', drop_first=True)
    if not county_dummies.empty:
        X = pd.concat([X, county_dummies], axis=1)

    # Add constant
    X = sm.add_constant(X)

    y = df['AvgScore']

    # Fit OLS with robust standard errors (HC3)
    model = sm.OLS(y, X).fit(cov_type='HC3')

    return model


