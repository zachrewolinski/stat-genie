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

    # Ensure numeric columns are numeric (coerce errors to NaN)
    num_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'read', 'math']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with essential missing values (students, teachers, or scores)
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove impossible/zero values to avoid division errors
    df = df[(df['teachers'] > 0) & (df['students'] > 0)]

    # Primary independent variable: students per teacher
    df['StudentsPerTeacher'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computer resources standardized by student population
    # If 'computer' is missing, this will become NaN; it's kept as a control and can be dropped later by the model if necessary
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Rename/alias columns for clarity in modeling
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']
    df['CalWorks'] = df['calworks']
    df['LunchPct'] = df['lunch']
    df['EnglishPct'] = df['english']

    # Keep columns that will be used in the statistical model (plus some original vars for diagnostics)
    cols_to_keep = [
        'StudentsPerTeacher', 'AvgScore', 'ComputerPerStudent', 'Expenditure',
        'Income', 'CalWorks', 'LunchPct', 'EnglishPct', 'grades', 'county',
        'students', 'teachers', 'read', 'math'
    ]

    # Some datasets may not have all columns; keep present ones
    cols_present = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_present]

    # Optionally: drop rows with any remaining missing values among model variables
    # (this is a conservative choice; depending on analysis goals, one might impute instead)
    df = df.dropna(subset=['StudentsPerTeacher', 'AvgScore'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    # Fit an OLS regression of average score on students-per-teacher and controls
    # Use county and grades as categorical fixed effects. Use robust (HC3) standard errors.
    import statsmodels.formula.api as smf

    # Ensure the dataframe passed in has the transformed columns
    required = ['AvgScore', 'StudentsPerTeacher']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Required column {r} not found in dataframe")

    # Build formula. Include county and grades as categorical controls if present.
    formula_terms = [
        'StudentsPerTeacher',
        'Expenditure',
        'Income',
        'CalWorks',
        'LunchPct',
        'ComputerPerStudent',
        'EnglishPct',
        'students'
    ]

    # Keep only terms that are present in df
    terms_present = [t for t in formula_terms if t in df.columns]

    formula = 'AvgScore ~ ' + ' + '.join(terms_present)

    # Add categorical fixed effects if available
    if 'grades' in df.columns:
        formula += ' + C(grades)'
    if 'county' in df.columns:
        formula += ' + C(county)'

    # Fit OLS with robust standard errors (HC3)
    model_res = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Print a concise result summary for quick inspection
    print(model_res.summary())

    # Return the fitted results object for further inspection by the caller
    return model_res


