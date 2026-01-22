from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/replace_and_positive_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the variables used in the modeling stage.

    Produces the following new columns used by the model:
      - StudentTeacherRatio: students / teachers
      - z_StudentTeacherRatio: standardized StudentTeacherRatio (z-score)
      - AvgScore: mean of read and math scores
      - ComputerPerStudent: computer / students
      - ln_students: natural log of students

    Drops rows with missing or invalid values in key variables.
    """
    df = df.copy()

    # Ensure numeric columns are present and drop rows missing critical outcome or numerator/denominator
    required_cols = ['read', 'math', 'students', 'teachers', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'computer']
    df = df.dropna(subset=required_cols)

    # Remove observations with non-positive teacher counts to avoid division by zero
    df = df[df['teachers'] > 0]

    # Create student-teacher ratio and outcome
    df['StudentTeacherRatio'] = df['students'] / df['teachers']
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Technology access per student
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Log of students to control for scale (avoid log(0) because students > 0)
    df['ln_students'] = np.log(df['students'])

    # Standardize the key independent variable for easier interpretation of coefficients
    mean_str = df['StudentTeacherRatio'].mean()
    std_str = df['StudentTeacherRatio'].std()
    # If std is 0 (unlikely), avoid division by zero
    if std_str == 0 or np.isnan(std_str):
        df['z_StudentTeacherRatio'] = 0.0
    else:
        df['z_StudentTeacherRatio'] = (df['StudentTeacherRatio'] - mean_str) / std_str

    # Ensure categorical controls are typed as category
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Final drop of any rows with NaNs in modeling columns (safe-guard)
    model_cols = ['z_StudentTeacherRatio', 'AvgScore', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputerPerStudent', 'ln_students', 'grades', 'county']
    df = df.dropna(subset=model_cols)

    # Reset index for a clean dataframe
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Run an OLS regression of AvgScore on standardized student-teacher ratio and controls.

    Formula:
      AvgScore ~ z_StudentTeacherRatio + expenditure + income + calworks + lunch + english
                 + ComputerPerStudent + ln_students + C(grades) + C(county)

    County and grades are included as categorical fixed effects via C(...).
    Robust standard errors (HC3) are used to protect against heteroskedasticity.

    Returns:
      - results: fitted statsmodels regression results object
    """
    import statsmodels.formula.api as smf

    # Specify the formula including categorical controls for grades and county
    formula = (
        'AvgScore ~ z_StudentTeacherRatio + expenditure + income + calworks + lunch + english '
        '+ ComputerPerStudent + ln_students + C(grades) + C(county)'
    )

    model = smf.ols(formula=formula, data=df)
    results = model.fit(cov_type='HC3')

    # Print a brief summary for quick inspection
    print(results.summary())

    return results


