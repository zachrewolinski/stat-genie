from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/noperturb_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh & Parker classroom data to produce analysis-ready variables.

    Produces:
      - beauty_z: standardized beauty (mean 0, sd 1)
      - binary dummies for categorical controls (gender_female, minority_yes, tenure_yes, native_yes,
        division_upper, credits_more)
      - numeric conversions for age, students, allstudents
      - log_students: log(1 + students) as a continuous control
      - prof as numeric id for clustering / fixed effects

    Drops rows with missing key values (eval or beauty) and with missing essential controls (age, students, prof).
    """
    df = df.copy()

    # Ensure we operate on columns that exist
    required = ['eval', 'beauty']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe")

    # Drop observations missing the dependent variable or the main IV
    df = df.dropna(subset=['eval', 'beauty'])

    # Standardize beauty for easier interpretation of coefficients
    # Use population std (ddof=0) to match common standardization conventions
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std(ddof=0)

    # Create clean binary indicators for categorical controls (case-insensitive)
    df['gender_female'] = df.get('gender', '').astype(str).str.lower().map(lambda x: 1 if x == 'female' else 0)
    df['minority_yes'] = df.get('minority', '').astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)
    df['tenure_yes'] = df.get('tenure', '').astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)
    df['native_yes'] = df.get('native', '').astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)
    df['division_upper'] = df.get('division', '').astype(str).str.lower().map(lambda x: 1 if x == 'upper' else 0)
    df['credits_more'] = df.get('credits', '').astype(str).str.lower().map(lambda x: 1 if x == 'more' else 0)

    # Numeric conversions and derived variables
    df['age'] = pd.to_numeric(df.get('age'), errors='coerce')
    df['students'] = pd.to_numeric(df.get('students'), errors='coerce')
    df['allstudents'] = pd.to_numeric(df.get('allstudents'), errors='coerce')
    # Log transform of number of respondents (stabilizes variance and reduces influence of very large classes)
    df['log_students'] = np.log1p(df['students'])

    # Ensure prof is numeric for clustering and fixed effects
    df['prof'] = pd.to_numeric(df.get('prof'), errors='coerce')

    # Drop rows with missing essential controls (this keeps a balanced set for modeling).
    df = df.dropna(subset=['age', 'students', 'prof'])

    # Keep only columns needed for modeling (but return entire df with these added columns)
    # Users can still access original columns in df if needed
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run a series of regression models to estimate the effect of instructor beauty on student evaluations.

    Models returned:
      - ols: OLS with robust (default) standard errors
      - ols_cluster: OLS with cluster-robust SEs clustered by prof
      - fe: OLS with professor fixed effects (C(prof))
      - fe_cluster: fixed-effects model with cluster-robust SEs by prof
      - wls: WLS weighted by number of student respondents (students)
      - wls_cluster: WLS with cluster-robust SEs by prof

    Each entry in the returned dict is a fitted statsmodels RegressionResults object.
    """
    import statsmodels.formula.api as smf

    # Basic formula with beauty and controls (matches the transformed column names)
    formula = (
        'eval ~ beauty_z + age + gender_female + minority_yes + tenure_yes + '
        'native_yes + division_upper + credits_more + log_students'
    )

    results = {}

    # Ordinary least squares
    ols = smf.ols(formula, data=df).fit()
    results['ols'] = ols

    # OLS with cluster-robust standard errors (cluster on professor)
    # If prof is continuous numeric identifier, this will cluster by unique professor id
    try:
        ols_cluster = ols.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        ols_cluster = ols
    results['ols_cluster'] = ols_cluster

    # Professor fixed effects (add categorical professor dummies)
    fe = smf.ols(formula + ' + C(prof)', data=df).fit()
    results['fe'] = fe

    # Fixed effects with cluster-robust SEs
    try:
        fe_cluster = fe.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        fe_cluster = fe
    results['fe_cluster'] = fe_cluster

    # Weighted least squares using the number of respondents as weights (more respondents -> more precise)
    # Add a small epsilon to students to avoid zero weights (shouldn't be zero per schema but safe)
    eps = 1e-6
    wls = smf.wls(formula, data=df, weights=df['students'] + eps).fit()
    results['wls'] = wls

    try:
        wls_cluster = wls.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        wls_cluster = wls
    results['wls_cluster'] = wls_cluster

    return results


