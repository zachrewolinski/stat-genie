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
    Transform the raw district-level dataframe into a cleaned dataframe with the variables
    needed for the analysis.

    Produces the following columns (exact names used in the model):
      - StudentTeacherRatio: Enrollment (calworks) divided by Teachers (teachers)
      - AvgScore: mean of 'grades' and 'rownames' (district average reading and math scores)
      - ExpenditurePerStudent: from 'expenditure'
      - PctCalWorks: from 'income' (percent qualifying for CalWorks)
      - PctReducedLunch: from 'math' (percent qualifying for reduced-price lunch)
      - PctEngLearners: from 'district' (percent English learners)
      - NumComputers: from 'english' (number of computers)
      - SchoolSpan: from 'school' (categorical)
      - Enrollment, Teachers: raw enrollment and teacher counts (kept for inspection)
    """
    df = df.copy()

    # Ensure required columns exist; attempt to coerce numeric columns
    # and drop rows that cannot be used for the primary variables.
    # 'calworks' is treated as enrollment, 'teachers' as teacher FTE.
    numeric_cols = ['calworks', 'teachers', 'grades', 'rownames', 'expenditure', 'income', 'math', 'district', 'english']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows without enrollment, teacher counts, or test scores
    df = df.dropna(subset=['calworks', 'teachers', 'grades', 'rownames'])

    # Remove impossible teacher values (avoid division by zero)
    df = df[df['teachers'] > 0]

    # Create Enrollment and Teachers columns (clean numeric)
    df['Enrollment'] = df['calworks']
    df['Teachers'] = df['teachers']

    # Student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['Enrollment'] / df['Teachers']

    # Dependent variable: average of reading and math district scores
    df['AvgScore'] = df[['grades', 'rownames']].astype(float).mean(axis=1)

    # Controls (coerce numeric where possible)
    df['ExpenditurePerStudent'] = df['expenditure']
    df['PctCalWorks'] = df['income']
    df['PctReducedLunch'] = df['math']
    df['PctEngLearners'] = df['district']
    df['NumComputers'] = df['english']

    # Categorical control: grade-span / school type
    if 'school' in df.columns:
        df['SchoolSpan'] = df['school'].astype('category')
    else:
        df['SchoolSpan'] = pd.Categorical(['unknown'] * len(df))

    # Keep only the columns required for modeling and inspection
    result_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ExpenditurePerStudent', 'PctCalWorks',
        'PctReducedLunch', 'PctEngLearners', 'NumComputers', 'SchoolSpan', 'Enrollment', 'Teachers'
    ]

    # Some rows may still have NaNs in controls; keep them (statsmodels will drop rows with NaNs when fitting)
    return df[result_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model predicting average district test score from student-teacher ratio
    controlling for expenditures, socioeconomic and resource variables. Returns the
    fitted statsmodels regression results object (with robust standard errors).

    Model specification:
      AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PctCalWorks
                 + PctReducedLunch + PctEngLearners + NumComputers + SchoolSpan(dummies)

    Uses heteroskedasticity-robust (HC3) standard errors.
    """
    df = df.copy()

    # Create dummies for SchoolSpan (drop_first to avoid multicollinearity)
    if 'SchoolSpan' in df.columns:
        school_dummies = pd.get_dummies(df['SchoolSpan'], prefix='SchoolSpan', drop_first=True)
    else:
        school_dummies = pd.DataFrame(index=df.index)

    # Predictor matrix
    predictors = [
        'StudentTeacherRatio', 'ExpenditurePerStudent', 'PctCalWorks',
        'PctReducedLunch', 'PctEngLearners', 'NumComputers'
    ]

    X = df[predictors].join(school_dummies)
    X = sm.add_constant(X, has_constant='add')

    # Outcome
    y = df['AvgScore']

    # Drop rows with any missing values in X or y (statsmodels requires complete cases)
    complete_cases = X.join(y).dropna()
    X_cc = complete_cases.drop(columns=['AvgScore']) if 'AvgScore' in complete_cases.columns else complete_cases
    y_cc = complete_cases['AvgScore']

    # Fit OLS with robust standard errors (HC3)
    ols = sm.OLS(y_cc, X_cc)
    results = ols.fit(cov_type='HC3')

    # Return the fitted results object. The caller can inspect results.summary(), results.params, etc.
    return results


