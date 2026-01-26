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
    Transform the raw Hamermesh & Parker classroom dataset into analysis-ready form.

    The function will:
    - drop rows missing the primary outcome or primary predictors
    - create binary indicators for categorical controls
    - standardize the beauty rating (z-score) and create a squared term
    - create an interaction between beauty_z and gender_male to test moderation
    - log-transform student counts to reduce skew

    Returns the dataframe with columns used in the model.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = ['eval', 'beauty', 'students', 'gender', 'minority', 'tenure', 'native', 'credits', 'division', 'age', 'prof']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing outcome or primary predictor(s)
    df = df.dropna(subset=['eval', 'beauty'])

    # For controls, it's preferable to drop rows with missing critical covariates (students, gender, age)
    df = df.dropna(subset=['students', 'gender', 'age'])

    # Create binary/dummy variables for categorical controls
    # Map known factor codings (dataset uses 'yes'/'no', 'male'/'female', 'more'/'single', 'upper'/'lower')
    df['gender_male'] = (df['gender'].astype(str).str.lower() == 'male').astype(int)
    df['minority_yes'] = (df['minority'].astype(str).str.lower() == 'yes').astype(int)
    df['tenure_yes'] = (df['tenure'].astype(str).str.lower() == 'yes').astype(int)
    df['native_yes'] = (df['native'].astype(str).str.lower() == 'yes').astype(int)
    df['credits_more'] = (df['credits'].astype(str).str.lower() == 'more').astype(int)
    df['division_upper'] = (df['division'].astype(str).str.lower() == 'upper').astype(int)

    # Numeric transformations
    # Stabilize/skewed student counts with log(1 + students)
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['log_students'] = np.log1p(df['students'])

    # Standardize beauty (z-score) so coefficients are interpretable
    df['beauty'] = pd.to_numeric(df['beauty'], errors='coerce')
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    if pd.isnull(beauty_mean) or pd.isnull(beauty_std) or beauty_std == 0:
        raise ValueError('Invalid beauty column: cannot compute z-score')
    df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std
    df['beauty_z_sq'] = df['beauty_z'] ** 2

    # Interaction to test moderation (gender moderates beauty effect)
    df['beauty_z_x_gender_male'] = df['beauty_z'] * df['gender_male']

    # Ensure age numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Keep only rows with no missing values in model columns
    model_cols = ['eval', 'beauty_z', 'beauty_z_sq', 'beauty_z_x_gender_male',
                  'gender_male', 'age', 'minority_yes', 'tenure_yes', 'native_yes',
                  'credits_more', 'division_upper', 'log_students', 'prof']
    df = df.dropna(subset=model_cols)

    # Ensure professor id is present for clustering; keep as int if possible
    try:
        df['prof'] = pd.to_numeric(df['prof'], errors='coerce').astype('Int64')
    except Exception:
        # leave as original if conversion fails, but ensure no nulls
        if df['prof'].isnull().any():
            df = df[df['prof'].notnull()]

    # Return full dataframe (analysis will use the columns listed above)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit OLS regression to estimate effect of instructor beauty on student evaluations.

    Model specification:
      eval ~ beauty_z + beauty_z_sq + gender_male + beauty_z:gender_male
             + age + minority_yes + tenure_yes + native_yes
             + credits_more + division_upper + log_students

    We fit OLS and compute cluster-robust standard errors at the professor level (prof).

    Returns the fitted statsmodels regression results object.
    """
    # Select variables
    X_cols = [
        'beauty_z',
        'beauty_z_sq',
        'gender_male',
        'beauty_z_x_gender_male',
        'age',
        'minority_yes',
        'tenure_yes',
        'native_yes',
        'credits_more',
        'division_upper',
        'log_students'
    ]
    # Ensure columns exist
    missing = [c for c in X_cols + ['eval', 'prof'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing columns required for modeling: {missing}")

    X = df[X_cols]
    y = df['eval']

    # Add constant
    X = sm.add_constant(X)

    # Fit OLS
    ols_model = sm.OLS(y, X)

    # If professor id available, cluster standard errors by prof; otherwise use robust HC3
    try:
        # convert prof to numpy array for clustering
        groups = df['prof'].values
        results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': groups})
    except Exception:
        # fallback to heteroskedasticity-robust (HC3)
        results = ols_model.fit(cov_type='HC3')

    # Return the fitted results object (caller can inspect summary, params, conf_int, etc.)
    return results


