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

    # Required columns for analysis
    required = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'english', 'lunch', 'grades', 'county']

    # Drop rows missing core variables (students, teachers, scores)
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove rows with non-positive teacher counts (invalid for ratio)
    df = df[df['teachers'] > 0]

    # Student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computer resources normalized by students
    # If students is zero (shouldn't be), will produce inf; handle later
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Log enrollment to capture scale effects (use log1p to be safe)
    df['LogStudents'] = np.log1p(df['students'])

    # Ensure categorical variables are typed properly for modeling
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Replace infinite values and drop remaining rows with missing values in modeling columns
    df = df.replace([np.inf, -np.inf], np.nan)

    model_columns = ['StudentTeacherRatio', 'AvgScore', 'ComputerPerStudent', 'LogStudents', 'expenditure', 'income', 'english', 'lunch', 'grades', 'county']
    df = df.dropna(subset=model_columns)

    # Final dataframe ready for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Formula: AvgScore regressed on StudentTeacherRatio plus controls.
    # C(grades) and C(county) include categorical fixed effects for grade span and county.
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + english + lunch '
        '+ ComputerPerStudent + LogStudents + C(grades) + C(county)'
    )

    # Fit OLS with robust (HC1) standard errors
    results = smf.ols(formula=formula, data=df).fit(cov_type='HC1')

    # Return the fitted results object (statsmodels.regression.linear_model.RegressionResultsWrapper)
    return results


