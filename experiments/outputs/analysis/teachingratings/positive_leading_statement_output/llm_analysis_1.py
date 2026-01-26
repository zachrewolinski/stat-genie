from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/positive_leading_statement_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Hamermesh classroom dataset into analysis-ready dataframe.

    Produces the following columns required by the model:
      - eval: dependent variable (student course evaluation score)
      - beauty_z: standardized beauty rating (mean 0, sd 1)
      - beauty_z_sq: squared standardized beauty (nonlinear term)
      - gender_female, minority_yes, credits_single, division_lower, native_yes, tenure_yes: binary controls
      - age, students, allstudents: numeric controls
      - log_students: log(1 + students) to reduce skew
      - beauty_gender_interaction: beauty_z * gender_female (moderation term)
      - prof: professor id (used for clustered SE)
    """
    df = df.copy()

    # Drop rows missing the core variables
    df = df.dropna(subset=['beauty', 'eval'])

    # Ensure categorical strings are cleaned
    for col in ['gender', 'minority', 'credits', 'division', 'native', 'tenure']:
        if col in df.columns:
            # Some columns may already be categorical; convert to lowercase strings where applicable
            try:
                df[col] = df[col].astype(str).str.lower().str.strip()
            except Exception:
                pass

    # Standardize beauty
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std()
    df['beauty_z_sq'] = df['beauty_z'] ** 2

    # Binary control variables
    # Gender: female = 1, male = 0 (if unknown, set to NaN and drop later)
    df['gender_female'] = np.where(df['gender'] == 'female', 1,
                                   np.where(df['gender'] == 'male', 0, np.nan))

    # Minority: yes = 1, no = 0
    df['minority_yes'] = np.where(df['minority'] == 'yes', 1,
                                  np.where(df['minority'] == 'no', 0, np.nan))

    # Credits single-credit elective = 1
    df['credits_single'] = np.where(df['credits'] == 'single', 1,
                                    np.where(df['credits'] == 'more', 0, np.nan))

    # Division: lower = 1, upper = 0
    df['division_lower'] = np.where(df['division'] == 'lower', 1,
                                    np.where(df['division'] == 'upper', 0, np.nan))

    # Native English speaker yes = 1
    df['native_yes'] = np.where(df['native'] == 'yes', 1,
                                np.where(df['native'] == 'no', 0, np.nan))

    # Tenure track yes = 1
    df['tenure_yes'] = np.where(df['tenure'] == 'yes', 1,
                                np.where(df['tenure'] == 'no', 0, np.nan))

    # Numeric controls: age, students, allstudents
    # Ensure numeric types
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'students' in df.columns:
        df['students'] = pd.to_numeric(df['students'], errors='coerce')
    if 'allstudents' in df.columns:
        df['allstudents'] = pd.to_numeric(df['allstudents'], errors='coerce')

    # Log transform students to reduce skew; add a small constant (not necessary here since min students >=5, but safe)
    df['log_students'] = np.log(df['students'] + 1)

    # Interaction: beauty by female
    df['beauty_gender_interaction'] = df['beauty_z'] * df['gender_female']

    # Keep only rows with no missing values in model columns (drop rows with missing controls)
    required = [
        'eval', 'beauty_z', 'beauty_z_sq', 'gender_female', 'beauty_gender_interaction',
        'age', 'minority_yes', 'credits_single', 'division_lower', 'native_yes', 'tenure_yes',
        'students', 'log_students', 'allstudents', 'prof'
    ]
    # Some columns may not exist in some dataset variants; intersect
    required_existing = [c for c in required if c in df.columns]
    df = df.dropna(subset=required_existing)

    # Ensure professor id is integer for clustering
    if 'prof' in df.columns:
        try:
            df['prof'] = pd.to_numeric(df['prof'], errors='coerce').astype(int)
        except Exception:
            # leave as-is if conversion fails
            pass

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit OLS regression of student evaluation (eval) on instructor beauty and controls.

    Model specification:
      eval = beta0 + beta1*beauty_z + beta2*beauty_z_sq + beta3*gender_female
             + beta4*(beauty_z * gender_female) + controls + error

    Robust inference: cluster standard errors by professor id ('prof').

    Returns the fitted statsmodels results object.
    """
    # Select regressors - require that transform has produced these columns
    regressors = [
        'beauty_z', 'beauty_z_sq', 'gender_female', 'beauty_gender_interaction',
        'age', 'minority_yes', 'credits_single', 'division_lower', 'native_yes', 'tenure_yes',
        'log_students', 'allstudents'
    ]

    # Make sure regressors exist in df
    regressors = [r for r in regressors if r in df.columns]

    X = df[regressors]
    X = sm.add_constant(X, has_constant='add')
    y = df['eval']

    # Fit OLS with clustered SEs by professor if prof exists
    if 'prof' in df.columns:
        results = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
    else:
        # Fall back to robust (HC3) SEs if no prof id available
        results = sm.OLS(y, X).fit(cov_type='HC3')

    # Print a concise summary and return the full results object
    try:
        print(results.summary())
    except Exception:
        pass

    return results


