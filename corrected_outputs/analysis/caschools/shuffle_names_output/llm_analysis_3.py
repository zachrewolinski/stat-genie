from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/shuffle_names_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into a cleaned dataset containing the
    variables used in the statistical model. The function creates:
      - StudentsTotal (from 'calworks')
      - TeachersFTE (from 'teachers')
      - StudentTeacherRatio = StudentsTotal / TeachersFTE
      - AvgTestScore = mean of 'grades' and 'rownames'
      - ExpenditurePerStudent (from 'read')
      - PctReducedLunch (from 'math')
      - NumComputers (from 'english')
      - PctEnglishLearners (from 'district')
      - AvgIncome (from 'income')
      - LogStudentTeacherRatio (logged ratio, for optional nonlinearity checks)

    The function coerces relevant columns to numeric and drops rows with missing values
    in the main variables used for modeling.
    """
    # Work on a copy
    df = df.copy()

    # Coerce columns that should be numeric (some dataset descriptions are inconsistent)
    numeric_cols = [
        'calworks',    # interpreted as total students/enrollment
        'teachers',    # full-time equivalent teachers
        'grades',      # reading score (standardized)
        'rownames',    # math score (standardized)
        'read',        # described as expenditure per student in schema
        'math',        # described as pct reduced-price lunch in schema
        'english',     # described as number of computers in schema
        'district',    # described as pct English learners in schema
        'income'       # district average income
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived/renamed columns used in the model
    # Use 'calworks' as total enrollment / number of students
    if 'calworks' in df.columns:
        df['StudentsTotal'] = df['calworks']
    else:
        df['StudentsTotal'] = np.nan

    df['TeachersFTE'] = df['teachers'] if 'teachers' in df.columns else np.nan

    # Compute Student-Teacher ratio
    df['StudentTeacherRatio'] = df['StudentsTotal'] / df['TeachersFTE']

    # Dependent variable: take the mean of available standardized test scores
    score_cols = [c for c in ['grades', 'rownames'] if c in df.columns]
    if len(score_cols) == 0:
        df['AvgTestScore'] = np.nan
    else:
        df['AvgTestScore'] = df[score_cols].mean(axis=1)

    # Controls (map columns according to dataset descriptions)
    # Expenditure per student
    df['ExpenditurePerStudent'] = df['read'] if 'read' in df.columns else np.nan
    # Percent reduced-price lunch
    df['PctReducedLunch'] = df['math'] if 'math' in df.columns else np.nan
    # Number of computers (or computer-related measure)
    df['NumComputers'] = df['english'] if 'english' in df.columns else np.nan
    # Percent English learners
    df['PctEnglishLearners'] = df['district'] if 'district' in df.columns else np.nan
    # Average income
    df['AvgIncome'] = df['income'] if 'income' in df.columns else np.nan

    # Log transform of the ratio to capture nonlinearities if needed
    df['LogStudentTeacherRatio'] = np.where(df['StudentTeacherRatio'] > 0,
                                            np.log(df['StudentTeacherRatio']),
                                            np.nan)

    # Drop rows missing the key variables used for modeling
    required = [
        'StudentTeacherRatio',
        'AvgTestScore',
        'ExpenditurePerStudent',
        'PctReducedLunch',
        'PctEnglishLearners',
        'AvgIncome',
        'StudentsTotal'
    ]
    present_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=present_required)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a weighted linear model to estimate the association between student-teacher ratio
    and average test scores, controlling for district characteristics. We use WLS with
    district enrollment as weights to reflect precision of district means (larger districts
    provide more precise average scores). Robust (HC3) standard errors are returned.

    Model specification:
      AvgTestScore ~ StudentTeacherRatio + ExpenditurePerStudent + PctReducedLunch
                     + PctEnglishLearners + NumComputers + AvgIncome

    Returns the fitted statsmodels result object.
    """
    df = df.copy()

    # Define outcome and predictors
    y = df['AvgTestScore']
    X_cols = [
        'StudentTeacherRatio',
        'ExpenditurePerStudent',
        'PctReducedLunch',
        'PctEnglishLearners',
        'NumComputers',
        'AvgIncome'
    ]
    X = df[X_cols]
    X = sm.add_constant(X)

    # Use StudentsTotal as weights (use a floor of 1 to avoid zero weights)
    weights = df['StudentsTotal'].fillna(1).clip(lower=1)

    # Fit weighted least squares and request robust covariance (HC3)
    wls_model = sm.WLS(y, X, weights=weights)
    results = wls_model.fit(cov_type='HC3')

    # Attach a convenience summary string (optional) but return results object
    return results


