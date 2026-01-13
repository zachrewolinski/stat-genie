from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into the analysis-ready dataframe.

    Produces the following columns used in modeling (kept or created):
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of 'read' and 'math'
      - ComputersPerStudent: computer / students
      - keeps expenditure, income, calworks, lunch, english, students, grades, county

    Rows with missing values in required fields or with non-positive students/teachers are dropped.
    """
    df = df.copy()

    # Columns required for analysis
    required = [
        'students', 'teachers', 'read', 'math', 'computer',
        'expenditure', 'income', 'calworks', 'lunch', 'english',
        'grades', 'county'
    ]

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required)

    # Remove impossible/nonpositive values for students or teachers
    df = df[(df['students'] > 0) & (df['teachers'] > 0)]

    # Create student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Create outcome: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Create computers per student (resource control)
    # If 'computer' is 0, result will be 0; division safe because students > 0
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Ensure categorical variables have category dtype (keeps them usable by formula interface)
    try:
        df['grades'] = df['grades'].astype('category')
    except Exception:
        # if conversion fails, keep as-is
        pass
    try:
        df['county'] = df['county'].astype('category')
    except Exception:
        pass

    # Return the transformed dataframe (contains original columns plus derived columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model to estimate the association between student-teacher ratio and average test score.

    Model formula (controls included):
      AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english
                 + ComputersPerStudent + np.log(students) + C(grades) + C(county)

    Returns the fitted statsmodels regression results object (with robust HC3 standard errors).
    """
    import statsmodels.formula.api as smf
    import numpy as np

    # Work on a copy to avoid modifying original
    df = df.copy()

    # Ensure required columns exist
    needed = [
        'AvgScore', 'StudentTeacherRatio', 'expenditure', 'income', 'calworks',
        'lunch', 'english', 'ComputersPerStudent', 'students', 'grades', 'county'
    ]
    missing = [c for c in needed if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Build formula. Use log(students) to flexibly control for size.
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english '
        '+ ComputersPerStudent + np.log(students) + C(grades) + C(county)'
    )

    # Fit OLS with robust (HC3) standard errors
    fit = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted results object for downstream inspection
    return fit


