from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/noperturb_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist
    required = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'grades', 'county']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing key numeric inputs
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove impossible or zero teacher counts to avoid division by zero
    df = df[df['teachers'] > 0]

    # Compute student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgTestScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per teacher (proxy for technology resources in classrooms)
    # If computer is missing or teachers is zero we've already filtered; still guard
    df['ComputersPerTeacher'] = df['computer'] / df['teachers']

    # Create control columns with clear names used in modeling
    # Copy existing numeric controls into final column names
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']
    df['PctCalWorks'] = df['calworks']
    df['PctLunch'] = df['lunch']
    df['PctEnglishLearners'] = df['english']

    # Grades dummy: 1 if KK-08, 0 otherwise (KK-06 or other)
    # Ensure consistent string representation
    df['Grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)

    # Keep county as a categorical variable for optional fixed-effects
    df['County'] = df['county'].astype('category')

    # Drop rows with missing values in controls that we will include in the model
    control_cols = ['Expenditure', 'Income', 'PctCalWorks', 'PctLunch', 'PctEnglishLearners', 'ComputersPerTeacher']
    df = df.dropna(subset=control_cols)

    # Optional: trim extreme outliers in StudentTeacherRatio (e.g., extremely large due to data issues)
    # We'll keep rows within 1st and 99th percentile to avoid undue influence
    lower = df['StudentTeacherRatio'].quantile(0.01)
    upper = df['StudentTeacherRatio'].quantile(0.99)
    df = df[(df['StudentTeacherRatio'] >= lower) & (df['StudentTeacherRatio'] <= upper)]

    # Final check: ensure no remaining missing in modeling columns
    model_cols = ['AvgTestScore', 'StudentTeacherRatio'] + control_cols + ['Grades_KK08', 'County']
    df = df.dropna(subset=model_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits an OLS model estimating the association between student-teacher ratio and average test score,
    controlling for expenditure, income, poverty indicators, english learners, technology, and grade-span.

    The model uses robust (HC1) standard errors to guard against heteroskedasticity.
    Returns the fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    # Ensure dataframe is the transformed one
    required_cols = ['AvgTestScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'PctCalWorks', 'PctLunch', 'PctEnglishLearners', 'ComputersPerTeacher', 'Grades_KK08']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # Primary linear specification
    formula = (
        'AvgTestScore ~ StudentTeacherRatio + Expenditure + Income + PctCalWorks '
        '+ PctLunch + PctEnglishLearners + ComputersPerTeacher + Grades_KK08'
    )

    results = smf.ols(formula, data=df).fit(cov_type='HC1')

    return results


