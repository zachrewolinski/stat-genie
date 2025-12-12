from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/caschools/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.

    Produces the following columns used in the model:
      - Enrollment: from 'calworks' (interpreted as total enrollment)
      - Teachers: from 'teachers' (FTE count)
      - StudentTeacherRatio: Enrollment / Teachers
      - AvgScore: average of available standardized-score columns ('grades' and 'rownames')
      - ExpenditurePerStudent: from 'expenditure'
      - PctFreeLunch: from 'math' (dataset documentation suggests this column contains percent reduced-price lunch)
      - PctEnglishLearners: from 'district' (dataset documentation suggests this column contains percent English learners)
      - Income: from 'income'
      - Computers: from 'english' (interpreted as number of computers)

    Notes: the original schema has inconsistent/descriptive labels; mapping is chosen to best reflect likely intended variables.
    """
    df = df.copy()

    # Map / coerce source columns to the canonical names we will use
    # Enrollment (use 'calworks' as total enrollment)
    if 'calworks' in df.columns:
        df['Enrollment'] = pd.to_numeric(df['calworks'], errors='coerce')
    else:
        df['Enrollment'] = np.nan

    # Teachers (use 'teachers')
    if 'teachers' in df.columns:
        df['Teachers'] = pd.to_numeric(df['teachers'], errors='coerce')
    else:
        df['Teachers'] = np.nan

    # Expenditure per student
    if 'expenditure' in df.columns:
        df['ExpenditurePerStudent'] = pd.to_numeric(df['expenditure'], errors='coerce')
    else:
        df['ExpenditurePerStudent'] = np.nan

    # Percent free/reduced-price lunch (schema suggests 'math' contains this)
    if 'math' in df.columns:
        df['PctFreeLunch'] = pd.to_numeric(df['math'], errors='coerce')
    else:
        df['PctFreeLunch'] = np.nan

    # Percent English learners (schema suggests 'district' contains this)
    if 'district' in df.columns:
        df['PctEnglishLearners'] = pd.to_numeric(df['district'], errors='coerce')
    else:
        df['PctEnglishLearners'] = np.nan

    # Income (district average income)
    if 'income' in df.columns:
        df['Income'] = pd.to_numeric(df['income'], errors='coerce')
    else:
        df['Income'] = np.nan

    # Computers (use 'english' which in schema appears to be count of computers)
    if 'english' in df.columns:
        df['Computers'] = pd.to_numeric(df['english'], errors='coerce')
    else:
        df['Computers'] = np.nan

    # Standardized test scores: use 'grades' and 'rownames' (schema samples suggest these are scaled scores ~600-700)
    score_cols = []
    for c in ['grades', 'rownames']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            score_cols.append(c)

    # Create AvgScore as the mean of available score columns
    if len(score_cols) > 0:
        df['AvgScore'] = df[score_cols].mean(axis=1, skipna=True)
    else:
        df['AvgScore'] = np.nan

    # Compute StudentTeacherRatio: Enrollment divided by Teachers
    # Remove records with non-positive or missing teachers to avoid division by zero
    df['StudentTeacherRatio'] = np.nan
    mask_valid = df['Enrollment'].notna() & df['Teachers'].notna()
    mask_positive_teachers = mask_valid & (df['Teachers'] > 0)
    df.loc[mask_positive_teachers, 'StudentTeacherRatio'] = (
        df.loc[mask_positive_teachers, 'Enrollment'] / df.loc[mask_positive_teachers, 'Teachers']
    )

    # Keep only rows with necessary variables for modeling
    needed_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ExpenditurePerStudent',
        'PctFreeLunch', 'PctEnglishLearners', 'Income', 'Computers', 'Enrollment', 'Teachers'
    ]

    df = df[needed_cols].copy()

    # Drop rows with missing dependent variable or independent variable
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgScore'])

    # For modeling, it is useful to drop rows with no control information entirely missing
    # but allow some controls missing so model uses available cases (we'll drop any remaining rows with NA in controls)
    df = df.dropna(subset=['ExpenditurePerStudent', 'PctFreeLunch', 'PctEnglishLearners', 'Income'])

    # Optionally, winsorize or remove extreme StudentTeacherRatio values (extreme outliers can distort OLS)
    # We'll cap ratio at the 99.5th percentile to reduce influence of extreme outliers
    if not df['StudentTeacherRatio'].empty:
        upper = df['StudentTeacherRatio'].quantile(0.995)
        df.loc[df['StudentTeacherRatio'] > upper, 'StudentTeacherRatio'] = upper

    # Reset index and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression to test whether lower student-teacher ratio is associated
    with higher average academic performance, controlling for expenditure and
    student composition variables. Returns the fitted results object with robust SEs.

    Model specification:
      AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PctFreeLunch
                 + PctEnglishLearners + Income + Computers

    Interpretation: coefficient on StudentTeacherRatio indicates the change in AvgScore
    associated with a one-unit increase in students per teacher (we expect a negative
    coefficient if lower ratio -> higher performance).
    """
    df = df.copy()

    # Define outcome and predictors
    y = df['AvgScore']
    X = df[['StudentTeacherRatio', 'ExpenditurePerStudent', 'PctFreeLunch', 'PctEnglishLearners', 'Income', 'Computers']]

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust (HC3) standard errors
    model = sm.OLS(y, X, missing='drop')
    results = model.fit(cov_type='HC3')

    # Return the fitted results object (caller can print summary or access params)
    return results


