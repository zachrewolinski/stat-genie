from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/noperturb_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district dataframe to produce the derived columns used in the model.

    Produces the following new columns used in modeling:
      - avg_test_score: mean of 'read' and 'math'
      - student_teacher_ratio: students / teachers
      - computer_per_student: computer / students
      - grades_KK08: dummy (1 if grades == 'KK-08', else 0)
      - log_students: natural log of students (where students > 0)

    Drops rows with missing values in any of the model-relevant columns.
    """
    # copy to avoid modifying original
    df = df.copy()

    # Ensure numeric columns are numeric
    for col in ['students', 'teachers', 'computer', 'read', 'math', 'expenditure', 'income', 'calworks', 'lunch', 'english']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create dependent variable: average of read and math
    # If either read or math is missing, the mean will be NaN; we'll drop those rows later
    df['avg_test_score'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio
    # Avoid division by zero / invalid teachers values
    df['student_teacher_ratio'] = np.where(df['teachers'] > 0, df['students'] / df['teachers'], np.nan)

    # Computer per student
    df['computer_per_student'] = np.where(df['students'] > 0, df['computer'] / df['students'], np.nan)

    # Binary indicator for grades (KK-08 vs KK-06). Create explicit dummy; treat other values as NaN.
    df['grades_KK08'] = np.where(df['grades'] == 'KK-08', 1,
                                 np.where(df['grades'] == 'KK-06', 0, np.nan))

    # Log of total enrollment - only defined for positive student counts
    df['log_students'] = np.where(df['students'] > 0, np.log(df['students']), np.nan)

    # Keep columns required for modeling
    required_cols = [
        'avg_test_score',
        'student_teacher_ratio',
        'expenditure',
        'income',
        'calworks',
        'lunch',
        'english',
        'computer_per_student',
        'grades_KK08',
        'log_students'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Optionally: reset index for a clean dataframe
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of average test score on student-teacher ratio and controls.

    Model specification:
      avg_test_score = beta0 + beta1 * student_teacher_ratio + beta2 * expenditure + beta3 * income
                       + beta4 * calworks + beta5 * lunch + beta6 * english
                       + beta7 * computer_per_student + beta8 * grades_KK08 + beta9 * log_students + error

    Returns the fitted statsmodels RegressionResultsWrapper (with robust HC3 standard errors).
    """
    import statsmodels.api as sm

    # Make a shallow copy to avoid side effects
    dfm = df.copy()

    # Define outcome and predictors
    y = dfm['avg_test_score']

    X_cols = [
        'student_teacher_ratio',
        'expenditure',
        'income',
        'calworks',
        'lunch',
        'english',
        'computer_per_student',
        'grades_KK08',
        'log_students'
    ]

    X = dfm[X_cols]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit OLS with robust (HC3) standard errors to guard against heteroskedasticity
    model = sm.OLS(y, X).fit(cov_type='HC3')

    # Return the fitted model result object. The caller can inspect summary() or params, bse, etc.
    return model


