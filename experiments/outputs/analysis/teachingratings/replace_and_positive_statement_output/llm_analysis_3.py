from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/replace_and_positive_statement_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh classroom dataset into the analysis dataframe.

    Outputs (keeps/creates columns used by the model):
      - eval: dependent variable (kept)
      - beauty_z: standardized beauty score (mean 0, sd 1)
      - beauty_z2: squared beauty_z to capture nonlinearity
      - gender_female: 1 if female, 0 if male
      - age: numeric age (kept)
      - minority_yes: 1 if minority == 'yes', 0 if 'no'
      - tenure_yes: 1 if tenure == 'yes', 0 if 'no'
      - native_yes: 1 if native == 'yes', 0 if 'no'
      - credits_single: 1 if credits == 'single', 0 otherwise
      - division_upper: 1 if division == 'upper', 0 otherwise
      - students_z: standardized students count (mean 0, sd 1)
      - prof: professor id (kept for clustering)

    Missing values in key columns (eval, beauty) are dropped. Categorical mappings handle unexpected values by treating them as NaN and then filling with 0 where appropriate.
    """
    df = df.copy()

    # Drop rows with missing outcome or main predictor
    df = df.dropna(subset=['eval', 'beauty'])

    # Standardize beauty
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    if beauty_std == 0 or np.isnan(beauty_std):
        df['beauty_z'] = 0.0
    else:
        df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Quadratic term for potential non-linear relationship
    df['beauty_z2'] = df['beauty_z'] ** 2

    # Binary indicators for categorical controls
    # Map expected categories to 1/0; unknown values -> NaN -> fill with 0
    df['gender_female'] = df['gender'].map({'female': 1, 'male': 0})
    df['gender_female'] = df['gender_female'].fillna(0).astype(int)

    df['minority_yes'] = df['minority'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    df['tenure_yes'] = df['tenure'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    df['native_yes'] = df['native'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    df['credits_single'] = df['credits'].map({'single': 1, 'more': 0}).fillna(0).astype(int)
    df['division_upper'] = df['division'].map({'upper': 1, 'lower': 0}).fillna(0).astype(int)

    # Age: ensure numeric, drop or coerce non-numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Class size: standardize 'students' (number who participated in evaluation)
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    students_mean = df['students'].mean()
    students_std = df['students'].std(ddof=0)
    if students_std == 0 or np.isnan(students_std):
        df['students_z'] = 0.0
    else:
        df['students_z'] = (df['students'] - students_mean) / students_std

    # Keep professor id for clustering; ensure it's numeric
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Final: drop any rows that have become missing in required controls (if any)
    required_cols = ['eval', 'beauty_z', 'beauty_z2', 'gender_female', 'age', 'minority_yes',
                     'tenure_yes', 'native_yes', 'credits_single', 'division_upper', 'students_z', 'prof']
    df = df.dropna(subset=required_cols)

    # Return the dataframe with the required columns (and keep other columns if user wishes)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Estimate the effect of instructor beauty on student evaluations.

    Model: OLS regression of eval on standardized beauty (linear and quadratic) plus controls.
    Standard errors are clustered by professor ('prof') to account for non-independence across courses taught by the same instructor.

    Returns the fitted statsmodels regression results object with cluster-robust SEs applied.
    """
    import statsmodels.formula.api as smf

    # Formula: main effect of beauty (linear and quadratic) plus controls
    formula = (
        'eval ~ beauty_z + beauty_z2 + gender_female + age + minority_yes + '
        'tenure_yes + native_yes + credits_single + division_upper + students_z'
    )

    # Fit OLS with cluster-robust standard errors by prof
    ols_model = smf.ols(formula=formula, data=df)
    # statsmodels allows passing cov_type='cluster' to fit(); groups provided in cov_kwds
    results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Return the fitted results object. Caller can inspect .summary(), .params, .bse, etc.
    return results


