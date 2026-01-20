from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/replace_with_rvs_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Drop rows missing the core variables required for analyses
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove invalid or zero teacher/student counts to avoid division by zero
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: students per teacher
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Controls: rename / compute columns used in the model
    df['PctCalWorks'] = df['calworks']
    df['PctFreeLunch'] = df['lunch']
    df['ExpenditurePerStudent'] = df['expenditure']
    df['IncomeK'] = df['income']
    df['PctEnglishLearners'] = df['english']

    # Resource proxy: computers per student
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Size control: log enrollment
    df['LogEnrollment'] = np.log(df['students'])

    # County as categorical (string) for fixed effects in model
    df['County'] = df['county'].astype(str)

    # Select and return only columns required for modeling
    cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'PctCalWorks',
        'PctFreeLunch',
        'ExpenditurePerStudent',
        'IncomeK',
        'PctEnglishLearners',
        'ComputersPerStudent',
        'LogEnrollment',
        'County'
    ]

    # Some of these columns may still contain NA (e.g., computer was NA) -> drop remaining NA rows
    df = df[cols].dropna()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Formula: effect of student-teacher ratio on average test score, controlling for observables
    formula = (
        'AvgScore ~ StudentTeacherRatio + PctCalWorks + PctFreeLunch + '
        'ExpenditurePerStudent + IncomeK + PctEnglishLearners + ComputersPerStudent + LogEnrollment + C(County)'
    )

    # Fit OLS with robust (HC1) standard errors
    results = smf.ols(formula, data=df).fit(cov_type='HC1')

    # Return the fitted results object (caller can examine summary / params / conf_int)
    return results


