from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/positive_leading_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into the analysis-ready dataframe.

    Produces the following new columns used in the model:
    - StudentTeacherRatio: students / teachers (students per teacher)
    - AvgScore: mean of 'read' and 'math'
    - ComputersPerStudent: computer / students

    The function drops rows with missing key variables and winsorizes StudentTeacherRatio at the 1st and 99th percentiles to reduce influence of extreme outliers.
    """
    df = df.copy()

    # Drop rows missing the core variables needed for the analysis
    required = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=required)

    # Ensure numeric types
    for col in ['students', 'teachers', 'computer', 'read', 'math', 'expenditure', 'income', 'calworks', 'lunch', 'english']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Re-drop if conversion produced NaNs in core columns
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Compute dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: students per teacher (student-teacher ratio)
    # teachers should be > 0; drop or filter non-positive teacher counts
    df = df[df['teachers'] > 0]
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Computers per student (resource control)
    # Avoid division by zero: students>0 guaranteed from earlier drop
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Winsorize StudentTeacherRatio to reduce extreme influence (1st and 99th percentiles)
    lower = df['StudentTeacherRatio'].quantile(0.01)
    upper = df['StudentTeacherRatio'].quantile(0.99)
    df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=lower, upper=upper)

    # Optional: drop any remaining rows with NA in control variables used in the model
    controls = ['expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'grades', 'county']
    existing_controls = [c for c in controls if c in df.columns]
    if len(existing_controls) > 0:
        df = df.dropna(subset=existing_controls, how='any')

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model to estimate the association between student-teacher ratio and average test score.

    Model specification:
      AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english + ComputersPerStudent + C(grades) + C(county)

    - C(grades) and C(county) are included as categorical fixed effects.
    - Robust (HC3) standard errors are used to account for potential heteroskedasticity.

    Returns the fitted statsmodels regression result object.
    """
    import statsmodels.formula.api as smf

    # Verify required columns exist
    required_cols = ['AvgScore', 'StudentTeacherRatio']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Build formula dynamically to include available controls
    control_terms = []
    potential_controls = ['expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent']
    for c in potential_controls:
        if c in df.columns:
            control_terms.append(c)
    # Include categorical controls if present
    if 'grades' in df.columns:
        control_terms.append('C(grades)')
    if 'county' in df.columns:
        control_terms.append('C(county)')

    rhs = ' + '.join(['StudentTeacherRatio'] + control_terms)
    formula = f'AvgScore ~ {rhs}'

    # Fit OLS with robust standard errors (HC3)
    model = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object (caller can print summary or extract coefficients)
    return model


