from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/replace_with_rvs_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh & Parker classroom dataset into a dataframe with
    variables used in the regression models.

    Produces the following columns required by the model:
      - eval: dependent variable (kept as-is)
      - beauty_z, beauty_z_sq: standardized beauty and its square
      - age, students, allstudents: numeric controls
      - minority_yes, gender_female, credits_more, division_upper, native_yes, tenure_yes: binary dummies
      - prof: instructor id (kept for clustering)

    Rows with missing values in any of these columns are dropped.
    """
    df = df.copy()

    # Keep only required raw columns if present (avoid KeyError later)
    required_raw = ['beauty', 'eval', 'age', 'students', 'allstudents', 'minority', 'gender', 'credits', 'division', 'native', 'tenure', 'prof']
    missing_cols = [c for c in required_raw if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Input dataframe is missing required columns: {missing_cols}")

    # Drop rows missing the core columns beauty or eval
    df = df.dropna(subset=['beauty', 'eval'])

    # Ensure numeric columns are numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['allstudents'] = pd.to_numeric(df['allstudents'], errors='coerce')
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Binary encodings for categorical controls (map expected categories to 1/0). If unexpected values appear, they become NaN.
    df['minority_yes'] = df['minority'].map({'yes': 1, 'no': 0})
    df['gender_female'] = df['gender'].map({'female': 1, 'male': 0})
    df['credits_more'] = df['credits'].map({'more': 1, 'single': 0})
    df['division_upper'] = df['division'].map({'upper': 1, 'lower': 0})
    df['native_yes'] = df['native'].map({'yes': 1, 'no': 0})
    df['tenure_yes'] = df['tenure'].map({'yes': 1, 'no': 0})

    # Standardize beauty (z-score). Use population std (ddof=0) for interpretability; small-sample choice won't change conclusions materially.
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    if beauty_std == 0 or np.isnan(beauty_std):
        raise ValueError('beauty column has zero or NaN standard deviation; cannot standardize')
    df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std
    df['beauty_z_sq'] = df['beauty_z'] ** 2

    # Drop rows with missing values in any variable used by the model
    model_cols = ['eval', 'beauty_z', 'beauty_z_sq', 'age', 'students', 'allstudents',
                  'minority_yes', 'gender_female', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes', 'prof']
    df = df.dropna(subset=model_cols)

    # Reset index and return only relevant columns (keeps original columns too but ensures required ones exist)
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run OLS regressions to estimate the association between instructor beauty and
    student evaluations. Two specifications are returned:
      - model1: simple bivariate regression of eval on beauty_z (clustered SE by prof)
      - model2: full model with controls and quadratic beauty term (clustered SE by prof)

    Returns a dict with keys 'model1' and 'model2' containing statsmodels result objects
    with clustered robust covariance matrices (clustered by 'prof').
    """
    import statsmodels.api as sm

    # Ensure the dataframe has necessary columns
    required = ['eval', 'beauty_z', 'beauty_z_sq', 'age', 'students', 'allstudents',
                'minority_yes', 'gender_female', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe missing required columns for modeling: {missing}")

    df = df.copy()

    # Dependent and independent sets
    y = df['eval']

    # Model 1: beauty only (with intercept)
    X1 = sm.add_constant(df[['beauty_z']])
    ols1 = sm.OLS(y, X1).fit()
    # Clustered SE by professor
    model1_clust = ols1.get_robustcov_results(cov_type='cluster', groups=df['prof'])

    # Model 2: beauty (quadratic) + controls
    X2_cols = ['beauty_z', 'beauty_z_sq', 'age', 'students', 'allstudents',
               'minority_yes', 'gender_female', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes']
    X2 = sm.add_constant(df[X2_cols])
    ols2 = sm.OLS(y, X2).fit()
    model2_clust = ols2.get_robustcov_results(cov_type='cluster', groups=df['prof'])

    # Return fitted result objects (clustered) so callers can print summaries or extract coefficients.
    return {
        'model1': model1_clust,
        'model2': model2_clust
    }


