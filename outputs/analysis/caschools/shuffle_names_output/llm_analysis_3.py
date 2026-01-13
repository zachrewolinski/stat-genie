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
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric where appropriate. The provided schema has some mismatched descriptions,
    # but the following columns will be used based on their observed sample ranges:
    # - 'calworks' is used as total enrollment (students)
    # - 'teachers' is used as number of full-time-equivalent teachers
    # - 'grades' and 'rownames' are used as two test score measures (to be averaged)
    # - 'read' is used as expenditure per student
    # - 'math' is used as percent free/reduced-price lunch
    # - 'district' is used as percent English learners
    # - 'english' is used as number of computers
    # - 'school' is used as a categorical school-type variable

    # Convert to numeric where appropriate (coerce errors to NaN)
    num_cols = ['calworks', 'teachers', 'grades', 'rownames', 'read', 'math', 'district', 'english']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create dependent variable: average of the two score columns (if both exist). If one missing, use the available one.
    score_cols = [c for c in ['grades', 'rownames'] if c in df.columns]
    if len(score_cols) == 0:
        raise ValueError("No score columns ('grades' or 'rownames') present in dataframe for AvgScore calculation.")
    df['AvgScore'] = df[score_cols].mean(axis=1)

    # Avoid division by zero: replace non-positive teacher counts with NaN
    if 'teachers' not in df.columns or 'calworks' not in df.columns:
        raise ValueError("Required columns 'calworks' (enrollment) and 'teachers' (FTE) are not both present.")
    df.loc[df['teachers'] <= 0, 'teachers'] = np.nan

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['calworks'] / df['teachers']

    # Controls: create standardized control columns with informative names
    df['ExpenditurePerStudent'] = df['read']  # based on schema mapping
    df['PctFreeLunch'] = df['math']  # percent qualifying for reduced-price lunch
    df['PctEnglishLearners'] = df['district']  # percent English learners
    df['NumComputers'] = df['english']  # number of computers (district-level)

    # Categorical control: school type / grade span
    if 'school' in df.columns:
        df['SchoolType'] = df['school'].astype(str)
    else:
        df['SchoolType'] = 'unknown'

    # Keep only rows with non-missing values for model variables
    model_cols = ['AvgScore', 'StudentTeacherRatio', 'ExpenditurePerStudent', 'PctFreeLunch', 'PctEnglishLearners', 'NumComputers', 'SchoolType']
    # If some control columns are not present (unlikely given above checks), adapt accordingly
    present_model_cols = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=[c for c in present_model_cols if c != 'SchoolType'])

    # Optional: trim extreme outliers in StudentTeacherRatio by winsorizing at 1st and 99th percentiles
    try:
        low = df['StudentTeacherRatio'].quantile(0.01)
        high = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=low, upper=high)
    except Exception:
        pass

    # Return dataframe containing at least the columns used in the model
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Fit an OLS regression of average test score on student-teacher ratio with controls.
    # Uses robust (HC3) standard errors to help with heteroskedasticity.
    import statsmodels.formula.api as smf

    # Ensure the transform has been run (expects columns named exactly as in cvars)
    required = ['AvgScore', 'StudentTeacherRatio', 'ExpenditurePerStudent', 'PctFreeLunch', 'PctEnglishLearners', 'NumComputers', 'SchoolType']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for model: {missing}")

    formula = ('AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PctFreeLunch + '
               'PctEnglishLearners + NumComputers + C(SchoolType)')

    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted results object (has .summary() for inspection)
    return model

