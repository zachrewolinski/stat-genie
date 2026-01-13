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
    Transform the raw district-level dataframe to create the variables needed for the analysis.

    Produces these new columns used in the model:
      - AvgScore: mean of 'read' and 'math'
      - StudentTeacherRatio: students / teachers
      - ComputersPerStudent: computer / students

    Also ensures categorical typing for 'grades' and 'county' and drops rows with missing values
    in variables required for the analysis.
    """
    df = df.copy()

    # Ensure required numeric columns are present and convert if necessary
    required_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'lunch', 'calworks', 'english', 'income', 'grades', 'county']
    # Drop rows missing core numeric or score variables
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Create dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Create independent variable: student-teacher ratio
    # Protect against division by zero just in case (teachers should be > 0)
    df['TeacherCountSafe'] = df['teachers'].replace({0: np.nan})
    df['StudentTeacherRatio'] = df['students'] / df['TeacherCountSafe']

    # Create computers per student control
    # If students==0 (shouldn't happen), set to NaN
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Cast categorical controls to category dtype
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Drop rows with missing values in constructed variables or key controls used in the model
    model_vars = ['AvgScore', 'StudentTeacherRatio', 'ComputersPerStudent', 'expenditure', 'lunch', 'calworks', 'english', 'income', 'grades', 'county']
    # Only drop those model vars that exist in the dataframe (to be robust)
    existing_model_vars = [v for v in model_vars if v in df.columns]
    df = df.dropna(subset=existing_model_vars)

    # Optionally: remove extreme outliers in ratio (e.g., top 0.5%) to avoid undue influence
    if 'StudentTeacherRatio' in df.columns:
        high_cut = df['StudentTeacherRatio'].quantile(0.995)
        low_cut = df['StudentTeacherRatio'].quantile(0.005)
        df = df[(df['StudentTeacherRatio'] <= high_cut) & (df['StudentTeacherRatio'] >= low_cut)]

    # Drop helper column
    if 'TeacherCountSafe' in df.columns:
        df = df.drop(columns=['TeacherCountSafe'])

    # Reset index for downstream modeling convenience
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Estimate the association between student-teacher ratio and average standardized test scores.

    Model specification (primary):
      AvgScore ~ StudentTeacherRatio + expenditure + lunch + calworks + english + income
                 + ComputersPerStudent + C(grades) + C(county)

    We use OLS with robust (HC3) standard errors to reduce sensitivity to heteroskedasticity.
    Returns the fitted regression results object.
    """
    import statsmodels.formula.api as smf

    # Build formula. Include categorical controls for grades and county (fixed effects).
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + lunch + calworks + english + income '
        '+ ComputersPerStudent + C(grades) + C(county)'
    )

    # Fit OLS with robust standard errors (HC3). The fit() call returns the results object.
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    return model


