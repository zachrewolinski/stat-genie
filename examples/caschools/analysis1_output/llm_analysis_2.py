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
    Transform the raw district dataset into a dataframe ready for modeling.

    Produces the following new columns used in the model:
      - AvgScore: mean of 'read' and 'math'
      - StudentTeacherRatio: students / teachers (NaN for non-positive teachers)
      - ComputersPerStudent: computer / students (NaN for students==0)
      - Expenditure, Income, CalWorks, Lunch, English, Students copied/renamed for clarity
      - Grades, County as categorical variables

    Drops rows with missing values in any of the model columns.
    """
    df = df.copy()

    # Create dependent variable: average of reading and math scores
    if not set(['read', 'math']).issubset(df.columns):
        raise KeyError("Input dataframe must contain 'read' and 'math' columns")
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Compute independent variable: student-teacher ratio
    # Protect against division by zero / invalid teacher counts
    df['StudentTeacherRatio'] = df['students'] / df['teachers']
    df.loc[~np.isfinite(df['StudentTeacherRatio']), 'StudentTeacherRatio'] = np.nan

    # Computers per student (resource intensity)
    df['ComputersPerStudent'] = df['computer'] / df['students']
    df.loc[~np.isfinite(df['ComputersPerStudent']), 'ComputersPerStudent'] = np.nan

    # Copy/rename controls to clear column names used in the model
    # Use the existing variables where available
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']
    df['CalWorks'] = df['calworks']
    df['Lunch'] = df['lunch']
    df['English'] = df['english']
    df['Students'] = df['students']

    # Categorical variables
    # Keep original categories but ensure dtype is categorical for modeling convenience
    if 'grades' in df.columns:
        df['Grades'] = df['grades'].astype('category')
    else:
        df['Grades'] = pd.Categorical([None] * len(df))

    if 'county' in df.columns:
        df['County'] = df['county'].astype('category')
    else:
        df['County'] = pd.Categorical([None] * len(df))

    # Define all columns required for the model and drop rows with missing values in these columns
    model_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'Expenditure',
        'Income',
        'CalWorks',
        'Lunch',
        'English',
        'ComputersPerStudent',
        'Students',
        'Grades',
        'County'
    ]

    # Ensure columns exist before dropna
    missing_cols = [c for c in model_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for modeling after transform: {missing_cols}")

    # Drop rows with any missing values in the model columns
    df = df.dropna(subset=model_cols).reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of average test score on student-teacher ratio and controls.

    Model specification:
      AvgScore ~ StudentTeacherRatio + Expenditure + Income + CalWorks + Lunch + English
                 + ComputersPerStudent + Students + C(Grades) + C(County)

    Returns a statsmodels results object with robust (HC3) standard errors.
    """
    import statsmodels.formula.api as smf

    # Ensure the transformed dataframe contains the expected columns
    required = [
        'AvgScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'CalWorks', 'Lunch',
        'English', 'ComputersPerStudent', 'Students', 'Grades', 'County'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Specify the formula. Grades and County are treated as categorical fixed effects.
    formula = (
        'AvgScore ~ StudentTeacherRatio + Expenditure + Income + CalWorks + Lunch + English '
        '+ ComputersPerStudent + Students + C(Grades) + C(County)'
    )

    ols = smf.ols(formula=formula, data=df).fit()

    # Return results with robust standard errors (HC3)
    results = ols.get_robustcov_results(cov_type='HC3')
    return results


