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
    # Work on a copy
    df = df.copy()

    # Convert relevant columns to numeric where possible (coerce invalids to NaN)
    numeric_cols = ['calworks', 'teachers', 'expenditure', 'income', 'math', 'district', 'computer', 'grades', 'rownames']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derive dependent variable: average of the two provided test-score columns
    # Use mean across columns, skipping NA if one of the two is present
    if 'grades' in df.columns and 'rownames' in df.columns:
        df['AvgTestScore'] = df[['grades', 'rownames']].mean(axis=1)
    else:
        # If one of the expected score columns is missing, create AvgTestScore from whichever exists
        available_scores = [c for c in ['grades', 'rownames'] if c in df.columns]
        if len(available_scores) == 1:
            df['AvgTestScore'] = df[available_scores[0]]
        else:
            df['AvgTestScore'] = np.nan

    # Derive independent variable: student-teacher ratio
    # Use 'calworks' as total students (per schema) and 'teachers' as number of FTE teachers
    if 'calworks' in df.columns and 'teachers' in df.columns:
        # Avoid division by zero by coercing zero teachers to NaN
        df.loc[df['teachers'] == 0, 'teachers'] = np.nan
        df['StudentTeacherRatio'] = df['calworks'] / df['teachers']
    else:
        df['StudentTeacherRatio'] = np.nan

    # Control: log enrollment (to capture district size, add 1 to avoid log(0))
    if 'calworks' in df.columns:
        df['LogEnrollment'] = np.log(df['calworks'].replace(0, np.nan) + 1)
    else:
        df['LogEnrollment'] = np.nan

    # Create a binary indicator for school type (KK-08) if 'school' exists
    if 'school' in df.columns:
        # Ensure consistent string comparison
        df['school'] = df['school'].astype(str)
        df['school_KK08'] = (df['school'] == 'KK-08').astype(float)
    else:
        df['school_KK08'] = np.nan

    # At minimum require the key modeling variables to be present: StudentTeacherRatio and AvgTestScore
    required = ['StudentTeacherRatio', 'AvgTestScore', 'expenditure', 'income', 'math', 'district', 'computer', 'LogEnrollment']
    existing_required = [c for c in required if c in df.columns]

    # Drop rows missing the primary IV or DV or any of the required controls that we will include in the model
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgTestScore'] + existing_required)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Prepare data for modeling. Expect transform() has been run already.
    # Define predictors to include in the OLS model
    predictor_cols = [
        'StudentTeacherRatio',
        'expenditure',    # expenditure per student
        'income',         # socio-economic indicator
        'math',           # percent reduced-price lunch
        'district',       # percent English learners
        'computer',       # number of computers
        'LogEnrollment',  # log enrollment
        'school_KK08'     # school type indicator
    ]

    # Keep only columns present in df to avoid KeyErrors
    predictor_cols = [c for c in predictor_cols if c in df.columns]

    # Define X and y
    X = df[predictor_cols].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['AvgTestScore']

    # Fit OLS with robust (HC3) standard errors to protect against heteroskedasticity
    model = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

    # Return the fitted results object
    return model


