from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/replace_with_rvs_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Required raw columns: 'students', 'teachers', 'read', 'math', 'computer'
    # Drop rows with missing core variables needed to construct IV and DV
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove impossible or zero values that would break ratios/logs
    df = df[df['students'] > 0]
    df = df[df['teachers'] > 0]

    # Dependent variable: average of reading and math scores
    df['AvgTestScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Control: computers per student (handle zero/NA computers)
    # If 'computer' is missing, will remain NaN; keep those rows only if other covariates present
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Control: log of students (district size)
    # Use natural log to reduce skew; students > 0 already enforced
    df['LogStudents'] = np.log(df['students'])

    # Keep the categorical columns 'grades' and 'county' as-is for use with patsy/formula
    # Ensure they are treated as strings/categorical
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype(str)
    if 'county' in df.columns:
        df['county'] = df['county'].astype(str)

    # Optionally drop rows missing any control variables we will include in the model
    # We choose to drop rows missing the key numeric controls to ensure OLS uses complete cases
    control_vars = ['expenditure', 'lunch', 'english', 'income', 'ComputerPerStudent', 'LogStudents']
    present_controls = [c for c in control_vars if c in df.columns]
    if len(present_controls) > 0:
        df = df.dropna(subset=present_controls)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs an OLS regression of AvgTestScore on StudentTeacherRatio controlling for covariates.
    Returns the fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['AvgTestScore', 'StudentTeacherRatio', 'expenditure', 'lunch', 'english', 'income', 'ComputerPerStudent', 'LogStudents', 'grades', 'county']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Specify formula: main IV + controls + categorical controls for grades and county
    formula = (
        'AvgTestScore ~ StudentTeacherRatio + expenditure + lunch + english + income '
        '+ ComputerPerStudent + LogStudents + C(grades) + C(county)'
    )

    # Fit OLS with robust (HC3) standard errors
    ols_res = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted results object so caller can inspect coefficients, summary, etc.
    return ols_res


