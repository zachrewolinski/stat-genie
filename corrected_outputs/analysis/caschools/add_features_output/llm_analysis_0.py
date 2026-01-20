from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district dataframe to the analytic dataframe.
    Produces the following columns required for modeling:
      - StudentTeacherRatio: students / teachers
      - AvgTestScore: mean of 'read' and 'math'
      - ComputersPerStudent: computer / students
      - LogStudents: natural log of students
    Ensures required control columns exist; drops rows with missing key variables.
    """
    df = df.copy()

    # Ensure key columns are present; if not, create as NaN so dropna below will work
    expected_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'grades', 'county']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows with missing essential measurement variables
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Exclude nonpositive teacher counts (would cause division by zero or nonsensical ratios)
    df.loc[df['teachers'] <= 0, 'teachers'] = np.nan
    df = df.dropna(subset=['teachers'])

    # Compute student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgTestScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student (resource control). Keep as computers per student (can be small decimal).
    # If 'computer' is zero or missing it will naturally produce 0 or NaN.
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log of students to capture nonlinearity with district size
    # Use natural log; add small epsilon guard if students==0 (should not happen after dropping above)
    df['LogStudents'] = np.log(df['students'])

    # Keep the control columns with consistent names expected by the model
    df['expenditure'] = df['expenditure']
    df['income'] = df['income']
    df['english'] = df['english']
    df['lunch'] = df['lunch']
    df['calworks'] = df['calworks']
    df['grades'] = df['grades']
    df['county'] = df['county']

    # Drop rows with missing values in any model-relevant column
    model_cols = ['StudentTeacherRatio', 'AvgTestScore', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'ComputersPerStudent', 'LogStudents', 'grades', 'county']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Run an OLS regression of AvgTestScore on StudentTeacherRatio controlling for several district covariates
    and county and grade-span fixed effects. Returns the fitted results object.

    Model specification:
      AvgTestScore ~ StudentTeacherRatio + expenditure + income + english + lunch + calworks
                    + ComputersPerStudent + LogStudents + C(grades) + C(county)

    Uses heteroskedasticity-robust standard errors (HC3).
    """
    import statsmodels.formula.api as smf

    # Ensure the dataframe passed to the model call contains the transformed columns
    required = ['AvgTestScore', 'StudentTeacherRatio', 'expenditure', 'income', 'english', 'lunch', 'calworks', 'ComputersPerStudent', 'LogStudents', 'grades', 'county']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    formula = (
        'AvgTestScore ~ StudentTeacherRatio + expenditure + income + english + lunch + calworks '
        '+ ComputersPerStudent + LogStudents + C(grades) + C(county)'
    )

    results = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    return results


