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
    Transform the raw Hamermesh & Parker classroom dataset into analysis-ready columns.

    Produces the following columns required by the model:
      - Eval: numeric dependent variable (same as original 'eval')
      - Beauty_z: standardized beauty rating (z-score)
      - Beauty_z_sq: squared z-score for nonlinearity
      - LogStudents: log of number of students who participated in evaluation
      - Age: instructor age
      - Female: binary indicator (1 female, 0 male)
      - Minority: binary indicator (1 minority, 0 not)
      - CreditsSingle: binary indicator (1 single-credit elective, 0 otherwise)
      - UpperDivision: binary indicator (1 upper division, 0 lower)
      - NativeSpeaker: binary indicator (1 native English speaker, 0 otherwise)
      - OnTenureTrack: binary indicator (1 yes, 0 no)
      - prof: professor id (kept for clustering)

    Notes: rows with missing data on core variables are dropped.
    """
    df = df.copy()

    # Core missingness: need eval and beauty at minimum; also students and age for controls
    df = df.dropna(subset=['eval', 'beauty', 'students', 'age'])

    # Dependent variable
    df['Eval'] = df['eval'].astype(float)

    # Standardize beauty (z-score) to make coefficient interpretable in SD units
    # use population-style std (ddof=0) to be explicit; either ddof=0 or ddof=1 is acceptable.
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    if beauty_std == 0 or np.isnan(beauty_std):
        df['Beauty_z'] = 0.0
    else:
        df['Beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Quadratic term to capture non-linear relationship
    df['Beauty_z_sq'] = df['Beauty_z'] ** 2

    # Class size control: log transform to reduce skew
    # students column is number of students who participated
    df['LogStudents'] = np.log(df['students'].astype(float) + 1e-9)

    # Age
    df['Age'] = df['age'].astype(float)

    # Binary encodings for categorical controls (map expected category labels to 0/1)
    df['Female'] = df['gender'].map({'female': 1, 'male': 0})
    # If gender has other encodings or missing, coerce to 0/1 with fillna
    df['Female'] = df['Female'].fillna(0).astype(int)

    df['Minority'] = df['minority'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    df['CreditsSingle'] = df['credits'].map({'single': 1, 'more': 0}).fillna(0).astype(int)
    df['UpperDivision'] = df['division'].map({'upper': 1, 'lower': 0}).fillna(0).astype(int)
    df['NativeSpeaker'] = df['native'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    df['OnTenureTrack'] = df['tenure'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # Preserve professor id for clustering
    df['prof'] = df['prof']

    # Return dataframe including original columns plus engineered ones
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Estimate effect of instructor beauty on course evaluations.

    Model specification: OLS with clustered standard errors at the professor level.
    Formula:
      Eval ~ Beauty_z + Beauty_z_sq + LogStudents + Age + Female + Minority
             + CreditsSingle + UpperDivision + NativeSpeaker + OnTenureTrack

    Returns the fitted statsmodels regression results object (with clustered SEs).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required = ['Eval', 'Beauty_z', 'Beauty_z_sq', 'LogStudents', 'Age', 'Female',
                'Minority', 'CreditsSingle', 'UpperDivision', 'NativeSpeaker',
                'OnTenureTrack', 'prof']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula
    formula = (
        'Eval ~ Beauty_z + Beauty_z_sq + LogStudents + Age + Female + Minority '
        '+ CreditsSingle + UpperDivision + NativeSpeaker + OnTenureTrack'
    )

    # Fit OLS and compute cluster-robust standard errors by professor
    model_fit = smf.ols(formula=formula, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['prof']}
    )

    # Return the fitted results (user can call .summary() or inspect params, pvalues, etc.)
    return model_fit


