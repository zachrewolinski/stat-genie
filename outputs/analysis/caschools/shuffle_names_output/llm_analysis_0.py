from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/shuffle_names_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe to produce the columns required for modeling.

    Creates the following columns (exact names used in modeling):
      - Enrollment: numeric total enrollment (from 'calworks')
      - NumTeachers: numeric number of teachers (from 'teachers')
      - StudentTeacherRatio: Enrollment / NumTeachers
      - ExpenditurePerStudent: numeric (from 'read')
      - PctReducedLunch: numeric (from 'math')
      - PctEnglishLearners: numeric (from 'district')
      - NumComputers: numeric (from 'english')
      - ReadingScore: numeric (from 'grades')
      - MathScore: numeric (from 'rownames')
      - AvgScore: mean of ReadingScore and MathScore
      - SchoolType: categorical (from 'school')

    The function drops rows with missing values in the key variables used for modeling.
    """
    df = df.copy()

    # Map/convert columns to numeric where appropriate. Use to_numeric with coercion.
    df['Enrollment'] = pd.to_numeric(df.get('calworks'), errors='coerce')
    df['NumTeachers'] = pd.to_numeric(df.get('teachers'), errors='coerce')
    df['ExpenditurePerStudent'] = pd.to_numeric(df.get('read'), errors='coerce')
    df['PctReducedLunch'] = pd.to_numeric(df.get('math'), errors='coerce')
    df['PctEnglishLearners'] = pd.to_numeric(df.get('district'), errors='coerce')
    df['NumComputers'] = pd.to_numeric(df.get('english'), errors='coerce')
    df['ReadingScore'] = pd.to_numeric(df.get('grades'), errors='coerce')
    df['MathScore'] = pd.to_numeric(df.get('rownames'), errors='coerce')

    # School type as categorical
    if 'school' in df.columns:
        df['SchoolType'] = df['school'].astype('category')
    else:
        df['SchoolType'] = pd.Categorical([None] * len(df))

    # Compute student-teacher ratio (protect against division by zero)
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']
    df['StudentTeacherRatio'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Compute average score as mean of reading and math scores
    df['AvgScore'] = df[['ReadingScore', 'MathScore']].mean(axis=1)

    # Drop rows missing the key model variables
    required_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ExpenditurePerStudent',
        'PctReducedLunch', 'PctEnglishLearners', 'NumComputers'
    ]
    df = df.dropna(subset=required_cols)

    # Optionally winsorize extreme StudentTeacherRatio values at 1st/99th percentile to reduce leverage
    try:
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=lower, upper=upper)
    except Exception:
        pass

    # Keep only the columns needed for modeling (plus helpful diagnostics)
    keep_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ExpenditurePerStudent',
        'PctReducedLunch', 'PctEnglishLearners', 'NumComputers', 'SchoolType',
        'Enrollment', 'NumTeachers', 'ReadingScore', 'MathScore'
    ]
    # Some columns may not exist if original dataset is missing them; select intersection
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit an OLS model of AvgScore on StudentTeacherRatio controlling for resource and demographic variables.

    Model specification:
      AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PctReducedLunch +
                 PctEnglishLearners + NumComputers + C(SchoolType)

    Uses heteroskedasticity-robust (HC3) standard errors.

    Returns the fitted statsmodels RegressionResults object.
    """
    # Assume df is already transformed (result of transform()). If not, transform here.
    required = ['StudentTeacherRatio', 'AvgScore']
    if not set(required).issubset(df.columns):
        df = transform(df)

    # Build design matrix
    X_cols = [
        'StudentTeacherRatio', 'ExpenditurePerStudent', 'PctReducedLunch',
        'PctEnglishLearners', 'NumComputers'
    ]
    X = df[X_cols].copy()

    # Include categorical school type dummies if available
    if 'SchoolType' in df.columns:
        dummies = pd.get_dummies(df['SchoolType'], prefix='SchoolType', drop_first=True)
        if not dummies.empty:
            X = pd.concat([X, dummies], axis=1)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Dependent variable
    y = df['AvgScore']

    # Fit OLS with robust standard errors
    model_res = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted model object (user can call .summary() on it)
    return model_res


