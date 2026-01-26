from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh classroom dataset into a dataframe ready for modeling.

    Produces the following new columns used in the model:
      - beauty_z: standardized beauty score (mean 0, sd 1)
      - beauty_z_sq: squared standardized beauty (to capture nonlinearity)
      - gender_female, minority_yes, tenure_yes, native_yes, division_upper, credits_single: binary dummies
      - log_students: log-transformed number of respondents
      - weights: number of respondents (used as WLS weights)

    Drops rows missing any of the essential variables: beauty, eval, students, prof.
    """
    df = df.copy()

    # Ensure key columns exist
    required = ['beauty', 'eval', 'students', 'prof']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input dataframe")

    # Drop rows with missing essential values
    df = df.dropna(subset=required)

    # Standardize beauty
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std()
    # If std is zero (unlikely), avoid division by zero
    if beauty_std == 0 or np.isnan(beauty_std):
        df['beauty_z'] = 0.0
    else:
        df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Quadratic term
    df['beauty_z_sq'] = df['beauty_z'] ** 2

    # Binary dummies for categorical controls (convert safely to string first)
    df['gender_female'] = df['gender'].astype(str).str.lower().str.strip().eq('female').astype(int)
    df['minority_yes'] = df['minority'].astype(str).str.lower().str.strip().eq('yes').astype(int)
    df['tenure_yes'] = df['tenure'].astype(str).str.lower().str.strip().eq('yes').astype(int)
    df['native_yes'] = df['native'].astype(str).str.lower().str.strip().eq('yes').astype(int)
    df['division_upper'] = df['division'].astype(str).str.lower().str.strip().eq('upper').astype(int)
    df['credits_single'] = df['credits'].astype(str).str.lower().str.strip().eq('single').astype(int)

    # Numeric controls
    # Ensure age is numeric if present; leave as NaN if missing
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['age'] = np.nan

    # Class size and weights
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    # Remove rows with non-positive or missing students
    df = df[df['students'].notnull() & (df['students'] > 0)]
    df['log_students'] = np.log(df['students'])
    # Use number of student respondents as WLS weight (more respondents -> more precise mean)
    df['weights'] = df['students']

    # Keep the evaluation score as numeric
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')

    # Drop rows that still have missing values in the modeling columns
    model_cols = [
        'eval', 'beauty_z', 'beauty_z_sq', 'age', 'gender_female', 'minority_yes',
        'tenure_yes', 'native_yes', 'division_upper', 'credits_single', 'log_students', 'weights', 'prof'
    ]
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a weighted linear model testing the effect of instructor beauty on student evaluations,
    controlling for instructor and course covariates. Uses WLS with weights equal to the
    number of student respondents and clusters standard errors at the instructor (prof) level.

    Model specification (main):
      eval ~ beauty_z + beauty_z_sq + age + gender_female + minority_yes + tenure_yes +
             native_yes + division_upper + credits_single + log_students

    Returns the fitted statsmodels result object (with cluster-robust SEs).
    """
    # Columns used as regressors
    exog_cols = [
        'beauty_z', 'beauty_z_sq', 'age', 'gender_female', 'minority_yes',
        'tenure_yes', 'native_yes', 'division_upper', 'credits_single', 'log_students'
    ]

    # Ensure required columns exist
    for c in exog_cols + ['eval', 'weights', 'prof']:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in transformed dataframe")

    # Build design matrix
    X = df[exog_cols]
    X = sm.add_constant(X)
    y = df['eval']
    weights = df['weights']

    # Fit weighted least squares
    wls_model = sm.WLS(y, X, weights=weights)
    # Fit and compute cluster-robust standard errors clustered by professor id
    results = wls_model.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Print a concise summary for quick inspection
    print(results.summary())

    # Return the fitted result object for downstream inspection
    return results


