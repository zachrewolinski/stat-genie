from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/shuffle_names_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh & Parker classroom dataset into a modeling-ready dataframe.

    Final columns produced (used in the model):
      - EvalScore: numeric teaching evaluation (from original 'tenure')
      - BeautyScore: raw continuous beauty score (from original 'prof')
      - Beauty_c: centered beauty score (BeautyScore - mean(BeautyScore))
      - Beauty_c2: squared centered beauty score
      - Female: binary indicator for female instructor (1 = female, 0 = male) from original 'age'
      - Age: numeric instructor age (from original 'division')
      - TenureTrack: binary indicator (1 = yes, 0 = no) from original 'eval'
      - NativeSpeaker: binary indicator (1 = yes, 0 = no) from original 'allstudents'
      - CourseLevelLower: binary indicator (1 = lower-division, 0 = upper-division) from original 'students'
      - ClassRespondents: numeric number of respondents (from original 'minority')
      - ClassEnrollment: numeric number enrolled (from original 'credits')
      - MinorityInstructor: binary indicator (1 = yes, 0 = no) from original 'beauty' (the factor that names 'yes'/'no')
      - InstructorID: instructor identifier (from original 'rownames')

    Notes/assumptions: column names in the provided schema are not fully consistent with their descriptions; the mapping implemented here follows typical usages in the published dataset (prof = continuous beauty; tenure = eval score). Rows with missing essential variables are dropped.
    """
    df = df.copy()

    # Ensure essential numeric columns are present and coerce types
    df['EvalScore'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['BeautyScore'] = pd.to_numeric(df['prof'], errors='coerce')

    # Map gender column: in this dataset 'age' contains 'male'/'female' strings
    df['Female'] = df['age'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})

    # Age: numeric; in original schema 'division' appears to hold numeric age values
    df['Age'] = pd.to_numeric(df['division'], errors='coerce')

    # Tenure track indicator: original 'eval' has 'yes'/'no'
    df['TenureTrack'] = df['eval'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Native English speaker indicator: original 'allstudents' used as yes/no for native
    df['NativeSpeaker'] = df['allstudents'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Course level: lower/upper
    df['CourseLevelLower'] = df['students'].astype(str).str.strip().str.lower().map({'lower': 1, 'upper': 0})

    # Class/respondent counts
    df['ClassRespondents'] = pd.to_numeric(df['minority'], errors='coerce')
    df['ClassEnrollment'] = pd.to_numeric(df['credits'], errors='coerce')

    # Minority instructor indicator: original 'beauty' field is a yes/no factor in the provided schema; map it as a control
    df['MinorityInstructor'] = df['beauty'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Instructor identifier
    df['InstructorID'] = df['rownames']

    # Center beauty score and add quadratic term to allow non-linear effect
    if df['BeautyScore'].notna().any():
        mean_beauty = df['BeautyScore'].mean()
    else:
        mean_beauty = 0.0
    df['Beauty_c'] = df['BeautyScore'] - mean_beauty
    df['Beauty_c2'] = df['Beauty_c'] ** 2

    # Drop rows missing the dependent variable or the primary IV
    required = ['EvalScore', 'Beauty_c']
    df = df.dropna(subset=required)

    # Optionally drop rows missing key controls to keep a consistent sample for OLS
    # (Users can relax this if they prefer listwise deletion with different choices)
    controls_for_listwise = ['Female', 'Age', 'TenureTrack', 'NativeSpeaker', 'ClassRespondents', 'ClassEnrollment']
    df = df.dropna(subset=controls_for_listwise)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model of EvalScore on centered beauty plus controls. Use robust (HC3) standard errors.

    Model specification:
      EvalScore ~ Beauty_c + Beauty_c2 + Female + Age + TenureTrack + NativeSpeaker
                 + ClassRespondents + ClassEnrollment + CourseLevelLower + MinorityInstructor
                 + (optionally) interaction Beauty_c * Female to test moderation by gender

    Returns the fitted statsmodels results object (with robust covariance).
    """
    import statsmodels.api as sm

    # Select predictors
    predictors = [
        'Beauty_c',
        'Beauty_c2',
        'Female',
        'Age',
        'TenureTrack',
        'NativeSpeaker',
        'ClassRespondents',
        'ClassEnrollment',
        'CourseLevelLower',
        'MinorityInstructor'
    ]

    # Build design matrix and outcome
    X = df[predictors].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['EvalScore'].astype(float)

    # Fit OLS with heteroskedasticity-robust standard errors (HC3)
    ols_mod = sm.OLS(y, X, missing='drop')
    results = ols_mod.fit(cov_type='HC3')

    # Optionally also fit an interaction model to test whether the beauty effect differs by gender
    # (commented out by default). To run, uncomment the following lines:
    # df['BeautyFemale'] = df['Beauty_c'] * df['Female']
    # X2 = df[['Beauty_c','BeautyFemale','Beauty_c2','Female','Age','TenureTrack','NativeSpeaker','ClassRespondents','ClassEnrollment','CourseLevelLower','MinorityInstructor']]
    # X2 = sm.add_constant(X2, has_constant='add')
    # interaction_results = sm.OLS(df['EvalScore'], X2, missing='drop').fit(cov_type='HC3')
    # print(interaction_results.summary())

    # Print summary for convenience
    print(results.summary())

    return results


