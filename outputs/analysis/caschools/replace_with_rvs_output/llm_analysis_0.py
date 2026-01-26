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
    """
    Transform the raw district data to variables used in the model.

    Outputs (added/modified columns):
    - StudentTeacherRatio: students / teachers
    - AvgScore: mean of read and math
    - ComputersPerStudent: computer / students
    - LogStudents: natural log of students
    - Drops rows with missing/invalid values for required columns
    """
    df = df.copy()

    # Ensure numeric columns are numeric where appropriate
    num_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'english', 'lunch', 'read', 'math']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing outcome or key inputs
    required_for_analysis = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=required_for_analysis)

    # Remove rows with non-positive teachers or students (to avoid division by zero)
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Create dependent variable: average of reading and math
    df['AvgScore'] = (df['read'] + df['math']) / 2.0

    # Create independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Computers per student (resource measure)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log of district size to control for scale
    df['LogStudents'] = np.log(df['students'].astype(float))

    # Keep relevant columns only (preserve original columns used as controls too)
    keep_cols = [
        'StudentTeacherRatio', 'AvgScore', 'ComputersPerStudent', 'LogStudents',
        'expenditure', 'income', 'english', 'lunch', 'county', 'grades',
        'students', 'teachers', 'computer', 'read', 'math'
    ]
    # Some columns may not exist in some datasets; intersect with actual columns
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df.loc[:, keep_cols].reset_index(drop=True)

    # Optionally winsorize extreme StudentTeacherRatio values at 1st and 99th percentiles
    try:
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower, upper)
    except Exception:
        pass

    # Impute missing control variables (expenditure, income, english, lunch, ComputersPerStudent) with median
    controls_to_impute = ['expenditure', 'income', 'english', 'lunch', 'ComputersPerStudent']
    for c in controls_to_impute:
        if c in df.columns:
            med = df[c].median()
            df[c] = df[c].fillna(med)

    # Ensure categorical controls are of type category
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for district resources and demographics.

    Model specification:
    AvgScore = beta0 + beta1*StudentTeacherRatio + beta2*expenditure + beta3*income + beta4*english
               + beta5*lunch + beta6*ComputersPerStudent + beta7*LogStudents + county FE + grades FE + error

    Returns the fitted statsmodels results object (with robust standard errors).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['AvgScore', 'StudentTeacherRatio']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Build formula. Include controls if present in df.
    controls = []
    for c in ['expenditure', 'income', 'english', 'lunch', 'ComputersPerStudent', 'LogStudents']:
        if c in df.columns:
            controls.append(c)

    # Add categorical controls
    cat_terms = []
    if 'county' in df.columns:
        cat_terms.append('C(county)')
    if 'grades' in df.columns:
        cat_terms.append('C(grades)')

    rhs_terms = ['StudentTeacherRatio'] + controls + cat_terms
    formula = 'AvgScore ~ ' + ' + '.join(rhs_terms)

    # Fit OLS with robust (HC3) standard errors
    model = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Print summary for quick inspection (caller can still use returned results)
    print(model.summary())

    return model


