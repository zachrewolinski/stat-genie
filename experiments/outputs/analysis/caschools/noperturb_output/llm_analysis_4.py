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
    Transform the raw districts dataframe for modeling.
    Produces the following new columns used in the model:
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of 'read' and 'math'
      - ComputersPerStudent: computer / students

    Drops rows with missing or invalid required values.
    """
    df = df.copy()

    # Required columns for our analysis
    required = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'english', 'lunch', 'computer', 'grades', 'county']

    # Drop rows missing any required column
    df = df.dropna(subset=required)

    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'english', 'lunch', 'computer']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows where conversions produced NaNs
    df = df.dropna(subset=numeric_cols)

    # Remove non-positive values that would invalidate ratios
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Derived variables
    df['StudentTeacherRatio'] = df['students'] / df['teachers']
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Convert categorical columns to category dtype for modeling with patsy/statsmodels
    df['grades'] = df['grades'].astype('category')
    df['county'] = df['county'].astype('category')

    # Reset index (optional) and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit an OLS regression of average academic performance on student-teacher ratio
    with controls for district resources and demographics. Returns the fitted model object.

    Model formula:
      AvgScore ~ StudentTeacherRatio + expenditure + income + english + lunch
                 + ComputersPerStudent + students + C(grades) + C(county)

    Robust (HC3) standard errors are used to make inference more robust to heteroskedasticity.
    """
    import statsmodels.formula.api as smf

    # Copy to avoid modifying caller's dataframe
    df = df.copy()

    # Ensure the dependent variable exists
    if 'AvgScore' not in df.columns or 'StudentTeacherRatio' not in df.columns:
        raise ValueError("Dataframe must contain 'AvgScore' and 'StudentTeacherRatio' columns. Run transform() first.")

    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + english + lunch '
        '+ ComputersPerStudent + students + C(grades) + C(county)'
    )

    model_results = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted results object (caller can inspect .summary(), .params, etc.)
    return model_results


