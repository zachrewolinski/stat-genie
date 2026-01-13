from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/noperturb_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = ['students', 'teachers', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'computer', 'read', 'math']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Basic filtering: drop rows missing the primary variables needed for the analysis
    required = ['students', 'teachers', 'read', 'math']
    # Also keep rows with key controls (expenditure, income, english, lunch, calworks, computer) if available
    required += ['expenditure', 'income', 'english', 'lunch', 'calworks', 'computer', 'grades', 'county']
    required_present = [c for c in required if c in df.columns]
    df = df.dropna(subset=required_present)

    # Avoid division by zero for teachers
    df.loc[df['teachers'] == 0, 'teachers'] = np.nan
    df = df.dropna(subset=['teachers', 'students'])

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Controls: copy/rename original variables to consistent column names used in the model
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']
    df['English'] = df['english']
    df['Lunch'] = df['lunch']
    df['CalWorks'] = df['calworks']

    # Computers per student (a per-pupil resource measure)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log of students to capture district size nonlinearly
    # Use natural log; students should be > 0 because we filtered earlier
    df['LogStudents'] = np.log(df['students'].astype(float))

    # Grades indicator: make binary flag for KK-08 (1) vs KK-06 (0)
    # Some datasets may use different capitalizations or NA; normalize to string
    df['grades'] = df['grades'].astype(str)
    df['Grades_KK08'] = df['grades'].apply(lambda x: 1 if x.strip() == 'KK-08' else 0)

    # County as-is (used for clustering). Ensure no missing counties remain
    df['County'] = df['county'].astype(str)

    # Keep only columns necessary for modeling to reduce size
    model_cols = [
        'AvgScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'English', 'Lunch', 'CalWorks',
        'ComputersPerStudent', 'LogStudents', 'Grades_KK08', 'County'
    ]
    # Only keep columns that exist (in case some controls were not in the dataset)
    model_cols = [c for c in model_cols if c in df.columns]

    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Ensure the required columns are present
    required = ['AvgScore', 'StudentTeacherRatio']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing from transformed dataframe: {c}")

    # Build design matrix with controls that are present
    candidate_controls = ['Expenditure', 'Income', 'English', 'Lunch', 'CalWorks', 'ComputersPerStudent', 'LogStudents', 'Grades_KK08']
    controls = [c for c in candidate_controls if c in df.columns]

    X_cols = ['StudentTeacherRatio'] + controls
    X = df[X_cols].astype(float)
    X = sm.add_constant(X)
    y = df['AvgScore'].astype(float)

    # Fit OLS with clustered standard errors at the county level if County is present
    if 'County' in df.columns:
        # statsmodels OLS and cluster covariance
        model = sm.OLS(y, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['County']})
    else:
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC1')  # heteroskedasticity-robust if no clustering

    # Return the fitted results object (contains params, summary, etc.)
    return results


