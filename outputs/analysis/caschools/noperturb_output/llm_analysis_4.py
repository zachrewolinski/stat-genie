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
    """
    Transform the raw district-level dataframe to include variables needed for modeling.
    Outputs columns used in the model: AvgScore, StudentTeacherRatio, ComputersPerStudent,
    and preserves controls: expenditure, income, english, lunch, calworks, grades, county.
    """
    df = df.copy()

    # Ensure numeric columns we will use are present; coerce errors to NaN
    numeric_cols = ['students', 'teachers', 'computer', 'read', 'math',
                    'expenditure', 'income', 'english', 'lunch', 'calworks']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the essential test score or size information
    df = df.dropna(subset=['read', 'math', 'students', 'teachers'])

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = (df['read'] + df['math']) / 2.0

    # Independent variable: student-teacher ratio (students per teacher)
    # Avoid division by zero or negative teacher counts
    df['StudentTeacherRatio'] = np.where(df['teachers'] > 0, df['students'] / df['teachers'], np.nan)

    # Control: computers per student (handle zero students)
    df['ComputersPerStudent'] = np.where(df['students'] > 0, df['computer'] / df['students'], np.nan)

    # Keep categorical control variables as-is but ensure consistent dtype
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    else:
        # create placeholder if missing
        df['grades'] = pd.Categorical(['Unknown'] * len(df))

    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')
    else:
        df['county'] = pd.Categorical(['Unknown'] * len(df))

    # Keep other controls; coerce to numeric if present
    for c in ['expenditure', 'income', 'english', 'lunch', 'calworks']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # After constructing derived columns, drop rows with missing key model variables
    required_for_model = ['AvgScore', 'StudentTeacherRatio', 'expenditure', 'income',
                          'english', 'lunch', 'calworks', 'ComputersPerStudent']
    # We will allow models to run even if some controls are missing by dropping only rows
    # missing dv or iv; but we'll also drop rows with missing the main controls to keep a common sample.
    present_required = [c for c in required_for_model if c in df.columns]
    # Always drop rows missing the DV or IV
    df = df.dropna(subset=['AvgScore', 'StudentTeacherRatio'])
    # Additionally drop rows missing the main resource control 'expenditure' when present
    if 'expenditure' in df.columns:
        df = df.dropna(subset=['expenditure'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Run an OLS regression of AvgScore on StudentTeacherRatio with controls.
    Includes county and grades as categorical fixed effects. Uses robust (HC3) standard errors.

    Returns the fitted statsmodels regression result object.
    """
    import statsmodels.formula.api as smf

    # Build formula: include main IV, controls, and categorical fixed effects
    # Note: C(grades) and C(county) let statsmodels create dummy variables for these factors
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + english + lunch + calworks '
        '+ ComputersPerStudent + C(grades) + C(county)'
    )

    # Fit OLS with robust standard errors
    model_fit = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    return model_fit


