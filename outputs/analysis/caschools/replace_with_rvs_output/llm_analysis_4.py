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
    # Work on a copy
    df = df.copy()

    # Ensure relevant numeric columns are numeric (coerce non-numeric to NaN)
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'english', 'lunch', 'calworks']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the core variables needed to compute ratio and outcome
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove impossible / degenerate values (zero or negative teachers or students)
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Create the student-teacher ratio (students per teacher)
    df['student_teacher_ratio'] = df['students'] / df['teachers']

    # Dependent variable: average of read and math scores
    df['avg_score'] = df[['read', 'math']].mean(axis=1)

    # Resource control: computers per student
    # If 'computer' missing, result will be NaN; we will impute below
    df['computers_per_student'] = df['computer'] / df['students']

    # Ensure categorical variables are proper dtype
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # For remaining numeric controls, impute a small number of missing values with the median
    controls_for_impute = ['expenditure', 'income', 'english', 'lunch', 'calworks', 'computers_per_student']
    for c in controls_for_impute:
        if c in df.columns:
            median_val = df[c].median()
            # If median is NaN (all missing), fill with 0 to avoid errors; otherwise use median
            if np.isnan(median_val):
                df[c] = df[c].fillna(0)
            else:
                df[c] = df[c].fillna(median_val)

    # Optional: Winsorize extreme student_teacher_ratio values to reduce influence of outliers
    # Here we cap to 1st and 99th percentiles
    if 'student_teacher_ratio' in df.columns:
        lower = df['student_teacher_ratio'].quantile(0.01)
        upper = df['student_teacher_ratio'].quantile(0.99)
        df['student_teacher_ratio'] = df['student_teacher_ratio'].clip(lower=lower, upper=upper)

    # Return dataframe containing all columns needed for modeling (and keep other original columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # This model estimates the association between student-teacher ratio and average test score,
    # controlling for district resources and demographics and including categorical fixed effects for grades and county.
    import statsmodels.formula.api as smf

    # Define formula: avg_score on main IV and controls. C() wraps categorical variables.
    formula = (
        'avg_score ~ student_teacher_ratio + expenditure + income + english + lunch + '
        'calworks + computers_per_student + C(grades) + C(county)'
    )

    # Fit OLS
    fit = smf.ols(formula=formula, data=df).fit()

    # Obtain robust (heteroskedasticity-consistent) standard errors (HC3)
    results = fit.get_robustcov_results(cov_type='HC3')

    # Print a concise summary and return the results object for further inspection
    print(results.summary())
    return results


