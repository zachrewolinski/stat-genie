from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/replace_and_positive_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into the analysis-ready dataframe.
    Creates:
      - AvgScore: mean of 'read' and 'math'
      - StudentTeacherRatio: students / teachers (drops rows with teachers <= 0)
      - ComputerPerStudent: computer / students
      - LogStudents: log(students)
      - Grades_KK08: indicator for 'KK-08'
    Drops rows with missing values in the variables required for the analysis.
    Returns the dataframe with all original columns plus the derived columns.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    num_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'read', 'math']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the two score components or student/teacher counts
    df = df.dropna(subset=['read', 'math', 'students', 'teachers'])

    # Remove invalid teacher counts (cannot divide by zero)
    df = df[df['teachers'] > 0]

    # Dependent variable: average of reading and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-to-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Resource controls
    # Computers per student (allow zero but avoid division by zero since students>0 here)
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Log of enrollment (students)
    df['LogStudents'] = np.log(df['students'].replace(0, np.nan))

    # Grades indicator: 1 if 'KK-08', 0 if 'KK-06' or other. Keep as numeric indicator.
    df['Grades_KK08'] = df['grades'].astype(str).apply(lambda x: 1 if x.strip() == 'KK-08' else 0)

    # Drop rows missing key control variables used in the main specification
    required_controls = ['expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputerPerStudent', 'LogStudents']
    # If any of those columns are missing entirely in the dataframe, skip the drop for that column
    present_required = [c for c in required_controls if c in df.columns]
    if len(present_required) > 0:
        df = df.dropna(subset=present_required)

    # Keep the county column (categorical) as-is for later expansion to dummies in the model
    # Final dataframe includes all original columns plus the derived columns used in the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs an OLS regression of AvgScore on StudentTeacherRatio controlling for resource and demographic covariates.
    Uses county fixed effects (dummies) and HC3 robust standard errors.

    Model specification:
      AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english
                 + ComputerPerStudent + LogStudents + Grades_KK08 + county fixed effects

    Returns:
      statsmodels regression results object (fitted model with robust SEs)
    """
    df = df.copy()

    # Ensure the necessary columns exist
    required_cols = ['AvgScore', 'StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputerPerStudent', 'LogStudents', 'Grades_KK08', 'county']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"The following required columns are missing from the dataframe: {missing}")

    # Build design matrix
    X_cols = ['StudentTeacherRatio', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputerPerStudent', 'LogStudents', 'Grades_KK08']
    X = df[X_cols].astype(float)

    # Create county dummies (drop first to avoid perfect multicollinearity)
    county_dummies = pd.get_dummies(df['county'].astype(str), prefix='county', drop_first=True)
    if not county_dummies.empty:
        X = pd.concat([X, county_dummies], axis=1)

    # Add constant
    X = sm.add_constant(X)

    y = df['AvgScore'].astype(float)

    # Fit OLS with robust (HC3) standard errors
    ols_mod = sm.OLS(y, X)
    ols_res = ols_mod.fit(cov_type='HC3')

    # Return the fitted results object so caller can inspect params, summary, etc.
    return ols_res


