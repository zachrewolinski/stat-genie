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
    Transform the raw district-level dataframe into a dataset with the exact columns
    used in the statistical model:
      - StudentTeacherRatio: Enrollment / Teachers, winsorized at 1st/99th percentiles
      - AvgScore: mean of reading and math average scores (columns 'grades' and 'rownames')
      - ExpenditurePerStudent: from 'expenditure'
      - PctReducedLunch: from 'math' (dataset metadata indicates this column is percent reduced-price lunch)
      - PctEnglishLearners: from 'district' (metadata indicates percent English learners)
      - Computers: from 'english' (metadata indicates number of computers)
      - Enrollment: from 'calworks' (metadata indicates total enrollment)
      - LogEnrollment: log1p(Enrollment)

    Notes about the dataset: the provided schema has some mismatched descriptions; the mapping used here
    follows the typical variables described in the dataset documentation (calworks=enrollment, teachers=FTE teachers,
    math=% reduced lunch, district=% English learners, english=# computers, expenditure=expenditure per student).
    """
    df = df.copy()

    # Ensure numeric types where appropriate
    numeric_cols = ['calworks', 'teachers', 'grades', 'rownames', 'expenditure', 'math', 'district', 'english']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the core variables needed to compute primary variables
    df = df.dropna(subset=['calworks', 'teachers', 'grades', 'rownames'])

    # Avoid division by zero or non-positive teacher counts
    df.loc[df['teachers'] <= 0, 'teachers'] = np.nan
    df = df.dropna(subset=['teachers'])

    # Create Enrollment and Teachers-based ratio
    df['Enrollment'] = df['calworks']
    df['StudentTeacherRatio'] = df['Enrollment'] / df['teachers']

    # Winsorize the StudentTeacherRatio at the 1st and 99th percentiles to reduce influence of extremes
    lower = df['StudentTeacherRatio'].quantile(0.01)
    upper = df['StudentTeacherRatio'].quantile(0.99)
    df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=lower, upper=upper)

    # Create the dependent variable: average of reading and math scores
    # According to the schema: 'grades' and 'rownames' contain average reading and math scores (names in schema are messy)
    df['AvgScore'] = df[['grades', 'rownames']].mean(axis=1)

    # Controls: map columns from the raw data to clear variable names
    # Expenditure per student
    if 'expenditure' in df.columns:
        df['ExpenditurePerStudent'] = df['expenditure']
    else:
        df['ExpenditurePerStudent'] = np.nan

    # Percent reduced-price lunch (proxy for poverty) - column 'math' per schema mapping
    if 'math' in df.columns:
        df['PctReducedLunch'] = df['math']
    else:
        df['PctReducedLunch'] = np.nan

    # Percent English learners - column 'district' per schema mapping
    if 'district' in df.columns:
        df['PctEnglishLearners'] = df['district']
    else:
        df['PctEnglishLearners'] = np.nan

    # Number of computers - column 'english' per schema mapping
    if 'english' in df.columns:
        df['Computers'] = df['english']
    else:
        df['Computers'] = np.nan

    # Log of enrollment to control for district size (use log1p to handle zeros safely)
    df['LogEnrollment'] = np.log1p(df['Enrollment'].clip(lower=0))

    # Keep only rows with non-missing DV and main IV and at least some controls (we will drop further in model)
    df = df.dropna(subset=['AvgScore', 'StudentTeacherRatio'])

    # Return the full dataframe with new columns (model will select required columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Run an OLS regression of AvgScore on StudentTeacherRatio and controls.
    Uses robust (HC3) standard errors to reduce sensitivity to heteroskedasticity.

    Model specification:
      AvgScore = beta0 + beta1 * StudentTeacherRatio + beta2 * ExpenditurePerStudent
                 + beta3 * PctReducedLunch + beta4 * PctEnglishLearners + beta5 * Computers
                 + beta6 * LogEnrollment + error

    Returns the fitted statsmodels results object.
    """
    # Select model variables and drop rows with missing values in any of them
    model_vars = [
        'AvgScore',
        'StudentTeacherRatio',
        'ExpenditurePerStudent',
        'PctReducedLunch',
        'PctEnglishLearners',
        'Computers',
        'LogEnrollment'
    ]
    df_model = df[model_vars].dropna()

    # Prepare X and y
    y = df_model['AvgScore']
    X = df_model.drop(columns=['AvgScore'])
    X = sm.add_constant(X)

    # Fit OLS with robust standard errors (HC3)
    ols_mod = sm.OLS(y, X)
    results = ols_mod.fit(cov_type='HC3')

    # Print a brief summary and return the results object
    print(results.summary())
    return results


