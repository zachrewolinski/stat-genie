from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/replace_with_rvs_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric (coerce bad values to NaN)
    numeric_cols = ['students', 'teachers', 'computer', 'read', 'math', 'expenditure', 'calworks', 'lunch', 'income', 'english']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing key outcome or denominator variables
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove rows with non-positive teachers or students to avoid invalid ratios
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Compute independent variable: students per teacher
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average score across reading and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student (control)
    # If 'computer' is total number of computers, normalize by students to get access per pupil
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Ensure categorical controls are treated as categories
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Drop rows with missing values in primary controls to keep a common sample
    control_cols = ['expenditure', 'calworks', 'lunch', 'income', 'english']
    present_controls = [c for c in control_cols if c in df.columns]
    if present_controls:
        df = df.dropna(subset=present_controls)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Formula: AvgScore explained by student-teacher ratio and controls, with categorical fixed effects for grades and county
    # Use heteroskedasticity-robust standard errors (HC3)
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + ComputersPerStudent + calworks + '
        'lunch + income + english + C(grades) + C(county)'
    )

    # Fit OLS with robust (HC3) standard errors
    results = smf.ols(formula, data=df).fit(cov_type='HC3')

    return results


