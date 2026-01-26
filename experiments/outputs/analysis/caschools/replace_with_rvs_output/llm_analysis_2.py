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
    Transform the raw district-level dataframe into a dataframe ready for modeling.

    Produces the following new/ensured columns used in the model:
      - AvgScore: mean of 'read' and 'math'
      - StudentTeacherRatio: students / teachers
      - ComputersPer100: (computer / students) * 100

    Keeps these existing columns (ensures dtype):
      - expenditure, lunch, english, income, students, grades, county
    """
    df = df.copy()

    # Ensure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = ['students', 'teachers', 'computer', 'expenditure', 'lunch', 'english', 'income', 'read', 'math']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the essential variables for the analysis
    required_for_analysis = ['students', 'teachers', 'read', 'math']
    missing_subset = [c for c in required_for_analysis if c in df.columns]
    df = df.dropna(subset=missing_subset)

    # Compute dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Compute Student-Teacher ratio (students per teacher)
    # Avoid division by zero -- replace nonpositive teachers with NaN before division
    df.loc[df['teachers'] <= 0, 'teachers'] = np.nan
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Compute Computers per 100 students
    # If students is zero or missing we will get NaN (already dropped large part above)
    df['ComputersPer100'] = df['computer'] / df['students'] * 100

    # Keep relevant control columns and ensure categorical columns are of appropriate dtype
    # grades and county may be factors; convert to string to be treated as categorical in formula
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype(str)
    if 'county' in df.columns:
        df['county'] = df['county'].astype(str)

    # Optionally drop rows with NaNs in controls that we will include in the regression
    control_cols = ['expenditure', 'lunch', 'english', 'income', 'ComputersPer100', 'students', 'grades', 'county']
    present_controls = [c for c in control_cols if c in df.columns]
    # It's acceptable to drop only rows with NaNs in the continuous controls to keep sample consistent
    continuous_controls = [c for c in ['expenditure', 'lunch', 'english', 'income', 'ComputersPer100', 'students'] if c in df.columns]
    if continuous_controls:
        df = df.dropna(subset=continuous_controls)

    # Optional: remove clearly implausible StudentTeacherRatio values (extreme outliers)
    # Clip to 1st-99th percentiles to reduce influence of extreme outliers, but keep most data
    if 'StudentTeacherRatio' in df.columns:
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=lower, upper=upper)

    # Final check: drop any remaining rows with NaNs in key model columns
    model_cols = ['AvgScore', 'StudentTeacherRatio'] + present_controls
    model_cols = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio and controls.

    Model specification:
      AvgScore ~ StudentTeacherRatio + expenditure + lunch + english + income + ComputersPer100 + students + C(grades) + C(county)

    Uses robust (HC3) standard errors to account for heteroskedasticity.

    Returns the fitted statsmodels regression results instance.
    """
    import statsmodels.formula.api as smf

    # Ensure the required columns are present
    required_cols = ['AvgScore', 'StudentTeacherRatio']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe")

    # Build formula. Include categorical fixed effects for grades and county if present.
    formula_parts = ['StudentTeacherRatio']
    continuous_controls = [c for c in ['expenditure', 'lunch', 'english', 'income', 'ComputersPer100', 'students'] if c in df.columns]
    formula_parts += continuous_controls

    # Add categorical controls (as factor variables) if available
    if 'grades' in df.columns:
        formula_parts.append('C(grades)')
    if 'county' in df.columns:
        formula_parts.append('C(county)')

    formula = 'AvgScore ~ ' + ' + '.join(formula_parts)

    # Fit OLS with robust standard errors (HC3)
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted model (results object). The user can call .summary() on it or inspect parameters.
    return model


