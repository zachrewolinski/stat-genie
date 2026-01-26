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
    Prepare data for analysis of the effect of instructor beauty on student evaluations.

    Outputs (appended to df):
      - beauty_z: standardized beauty score (mean 0, sd 1)
      - beauty_z_sq: squared standardized beauty (for optional nonlinearity checks)
      - gender_female, minority_yes, credits_more, division_upper, native_yes, tenure_yes: binary indicators (0/1)
      - ln_students: natural log of 'students'
      - eval: dependent variable (kept as-is)
      - prof: professor identifier (kept as-is)

    Rows with missing values in the key variables are dropped.
    """
    df = df.copy()

    # Required columns list
    required = ['beauty', 'eval', 'age', 'gender', 'students', 'minority', 'credits', 'division', 'native', 'tenure', 'prof']
    # Drop rows missing any required field
    df = df.dropna(subset=required)

    # Standardize beauty for interpretability
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / (df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1.0)
    # optional squared term in case of nonlinear effects
    df['beauty_z_sq'] = df['beauty_z'] ** 2

    # Binary indicators from categorical fields (explicit mappings to ensure reproducibility)
    df['gender_female'] = (df['gender'].astype(str).str.lower() == 'female').astype(int)
    df['minority_yes'] = (df['minority'].astype(str).str.lower() == 'yes').astype(int)
    df['credits_more'] = (df['credits'].astype(str).str.lower() == 'more').astype(int)
    df['division_upper'] = (df['division'].astype(str).str.lower() == 'upper').astype(int)
    df['native_yes'] = (df['native'].astype(str).str.lower() == 'yes').astype(int)
    df['tenure_yes'] = (df['tenure'].astype(str).str.lower() == 'yes').astype(int)

    # Log-transform number of students who participated to reduce skew
    # Add a small constant to avoid log(0) in pathological cases (there are no zeros per schema, but defensive coding)
    df['ln_students'] = np.log(df['students'].astype(float) + 1e-6)

    # Ensure eval is numeric
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')

    # Final drop for any newly coerced NaNs
    df = df.dropna(subset=['beauty_z', 'eval', 'age', 'ln_students', 'prof'])

    # Keep only columns necessary for modeling plus identifiers for traceability
    keep_cols = ['beauty', 'beauty_z', 'beauty_z_sq', 'eval', 'age', 'gender_female', 'minority_yes', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes', 'ln_students', 'prof']
    # If any keep_cols are missing (shouldn't be), create them as NA to avoid KeyError
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs two OLS specifications to estimate the effect of instructor beauty on student evaluations.

    1) Baseline OLS with controls and professor-clustered standard errors.
    2) OLS with professor fixed effects (C(prof)) and controls as a robustness check.

    Returns a dictionary with the fitted results objects:
      - 'clustered': OLS results with clustered SE by prof
      - 'fe': OLS results with professor fixed effects

    Usage: results = model(transformed_df)
    """
    import statsmodels.api as _sm
    import statsmodels.formula.api as _smf

    # Ensure required columns exist
    needed = ['beauty_z', 'eval', 'age', 'gender_female', 'minority_yes', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes', 'ln_students', 'prof']
    missing = [c for c in needed if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Baseline OLS with controls
    X_cols = ['beauty_z', 'age', 'gender_female', 'minority_yes', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes', 'ln_students']
    X = _sm.add_constant(df[X_cols])
    y = df['eval']

    model_cluster = _sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Robustness: professor fixed effects (lots of dummies) using formula API
    formula = 'eval ~ beauty_z + age + gender_female + minority_yes + credits_more + division_upper + native_yes + tenure_yes + ln_students + C(prof)'
    model_fe = _smf.ols(formula, data=df).fit()

    # Return both fitted models; callers can inspect .summary() or coefficients
    return {
        'clustered': model_cluster,
        'fe': model_fe
    }


