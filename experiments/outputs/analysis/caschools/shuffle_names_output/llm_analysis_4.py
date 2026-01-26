from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/shuffle_names_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe to create the variables used in the analysis.

    Assumptions / mappings (based on the provided schema descriptions):
    - 'calworks' contains total student enrollment (counts).
    - 'teachers' contains full-time-equivalent teacher counts.
    - 'grades' contains average reading score.
    - 'rownames' contains average math score.
    - 'read' contains expenditure per student.
    - 'math' contains percent qualifying for reduced-price lunch.
    - 'district' contains percent English learners.
    - 'english' contains number of computers (district-level).
    - 'county' and 'school' are categorical identifiers.

    The function will coerce relevant columns to numeric, compute the student-teacher ratio,
    create the dependent variable AvgTestScore, and return a dataframe with all variables
    needed for modeling.
    """
    df = df.copy()

    # Coerce mapped columns to numeric (errors -> NaN)
    df['total_students'] = pd.to_numeric(df.get('calworks', pd.Series()), errors='coerce')
    df['teachers_fte'] = pd.to_numeric(df.get('teachers', pd.Series()), errors='coerce')

    df['reading_score'] = pd.to_numeric(df.get('grades', pd.Series()), errors='coerce')
    df['math_score'] = pd.to_numeric(df.get('rownames', pd.Series()), errors='coerce')

    df['expenditure_per_student'] = pd.to_numeric(df.get('read', pd.Series()), errors='coerce')
    df['pct_reduced_lunch'] = pd.to_numeric(df.get('math', pd.Series()), errors='coerce')
    df['pct_english_learners'] = pd.to_numeric(df.get('district', pd.Series()), errors='coerce')
    df['num_computers'] = pd.to_numeric(df.get('english', pd.Series()), errors='coerce')

    # Create dependent variable: average of reading and math scores
    df['AvgTestScore'] = df[['reading_score', 'math_score']].mean(axis=1)

    # Drop rows missing the essential variables for ratio and outcome
    df = df.dropna(subset=['total_students', 'teachers_fte', 'AvgTestScore'])

    # Remove non-positive teacher counts to avoid division errors
    df = df[df['teachers_fte'] > 0]

    # Compute student-teacher ratio
    df['student_teacher_ratio'] = df['total_students'] / df['teachers_fte']

    # Create some logged versions (useful for diagnostics / alternate specifications)
    # Replace non-positive/zero expenditures with NaN before log
    df['log_student_teacher_ratio'] = np.log(df['student_teacher_ratio'].replace({0: np.nan}))
    df['log_expenditure_per_student'] = np.where(df['expenditure_per_student'] > 0,
                                                 np.log(df['expenditure_per_student']),
                                                 np.nan)

    # Ensure categorical controls are strings (safe for model formulas)
    if 'county' in df.columns:
        df['county'] = df['county'].astype(str)
    else:
        # create a placeholder if original column missing
        df['county'] = df.get('county', pd.Series(index=df.index, dtype='object')).astype(str)

    if 'school' in df.columns:
        df['school'] = df['school'].astype(str)
    else:
        df['school'] = df.get('school', pd.Series(index=df.index, dtype='object')).astype(str)

    # Final dataframe: keep only the columns needed for modeling (plus a few alternates)
    out_cols = [
        'student_teacher_ratio',
        'log_student_teacher_ratio',
        'AvgTestScore',
        'expenditure_per_student',
        'log_expenditure_per_student',
        'pct_reduced_lunch',
        'pct_english_learners',
        'num_computers',
        'county',
        'school'
    ]

    # Some of these may not exist in the original df; ensure they are present (if missing, create NaN column)
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and average test scores.

    Primary specification:
      AvgTestScore ~ student_teacher_ratio + expenditure_per_student + pct_reduced_lunch
                    + pct_english_learners + num_computers + C(school) + C(county)

    Robust standard errors (HC3) are used to account for heteroskedasticity.

    The function returns the fitted statsmodels regression results object.
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Drop rows with missing outcome or key predictors used in the formula
    formula = (
        'AvgTestScore ~ student_teacher_ratio + expenditure_per_student '
        '+ pct_reduced_lunch + pct_english_learners + num_computers + C(school) + C(county)'
    )

    # Drop rows with NaNs in variables that appear in the formula
    # Collect variable names from formula (basic parsing)
    required_vars = ['AvgTestScore', 'student_teacher_ratio', 'expenditure_per_student',
                     'pct_reduced_lunch', 'pct_english_learners', 'num_computers', 'school', 'county']
    df_model = df.dropna(subset=['AvgTestScore', 'student_teacher_ratio'])

    # It's acceptable if some controls are missing; the model will drop rows with missing values automatically
    # but ensure we at least attempt to include all columns referenced.

    # Fit OLS with heteroskedasticity-robust standard errors (HC3)
    results = smf.ols(formula, data=df_model).fit(cov_type='HC3')

    # Print a short summary to the console and return the results object
    print(results.summary())

    return results


