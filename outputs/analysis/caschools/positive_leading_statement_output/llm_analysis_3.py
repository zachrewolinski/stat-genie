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
    Transform the raw district-level dataframe into the variables required for modeling.

    Creates:
    - student_teacher_ratio: students / teachers (winsorized at 1st/99th percentile)
    - avg_score: mean of 'read' and 'math'
    - computers_per_student: computer / students
    - ensures 'grades' and 'county' are categorical

    Drops rows with missing or invalid values for core variables.
    """
    df = df.copy()

    # Keep rows that have the essential numeric inputs
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove invalid sizes/teacher counts
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Dependent variable: average of reading and math
    df['avg_score'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio (students per teacher)
    df['student_teacher_ratio'] = df['students'] / df['teachers']

    # Resource control: computers per student
    # Replace infinite values and small numerical issues
    df['computers_per_student'] = df['computer'] / df['students']
    df['computers_per_student'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Winsorize student_teacher_ratio to reduce influence of extreme outliers (1st/99th percentiles)
    if df['student_teacher_ratio'].notnull().any():
        lower = df['student_teacher_ratio'].quantile(0.01)
        upper = df['student_teacher_ratio'].quantile(0.99)
        df['student_teacher_ratio'] = df['student_teacher_ratio'].clip(lower, upper)

    # Ensure categorical variables are typed correctly
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Drop rows that became NA after computations
    df = df.dropna(subset=['avg_score', 'student_teacher_ratio', 'computers_per_student'])

    # Keep only columns that will be used in the model plus identifiers for traceability
    required_cols = [
        'student_teacher_ratio', 'avg_score', 'computers_per_student',
        'students', 'teachers', 'income', 'expenditure', 'calworks', 'lunch', 'english',
        'grades', 'county', 'district', 'school', 'rownames'
    ]

    # Some columns may not exist in every input; intersect
    present = [c for c in required_cols if c in df.columns]
    df = df.loc[:, present]

    # Return transformed dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating the association between student-teacher ratio and academic performance.

    Model specification:
      avg_score ~ student_teacher_ratio + income + expenditure + calworks + lunch + english
                  + computers_per_student + students + C(grades) + C(county)

    Uses robust (HC3) standard errors to guard against heteroskedasticity.

    Returns the fitted statsmodels regression results object.
    """
    import statsmodels.formula.api as smf

    # Make a working copy and drop rows missing any modeling variables
    model_df = df.copy()
    needed = [
        'avg_score', 'student_teacher_ratio', 'income', 'expenditure', 'calworks',
        'lunch', 'english', 'computers_per_student', 'students', 'grades', 'county'
    ]
    missing = [c for c in needed if c not in model_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    model_df = model_df.dropna(subset=needed)

    # Define formula with categorical indicators for grades and county
    formula = (
        'avg_score ~ student_teacher_ratio + income + expenditure + calworks + lunch '
        '+ english + computers_per_student + students + C(grades) + C(county)'
    )

    # Fit OLS with robust standard errors (HC3)
    results = smf.ols(formula=formula, data=model_df).fit(cov_type='HC3')

    # Print a concise summary and return full results object
    print(results.summary())
    return results


