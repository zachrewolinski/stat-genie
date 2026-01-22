from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/shuffle_names_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh & Parker classroom dataset into an analysis-ready dataframe.

    Outputs (columns required by the model):
    - EvalScore: numeric evaluation score (from column 'tenure').
    - BeautyScore: numeric continuous beauty rating (from column 'prof').
    - BeautyScore_c: mean-centered BeautyScore.
    - BeautyBinary: (optional) binary indicator from 'beauty' if present (yes=1/no=0).
    - Female: gender indicator (1 if 'female' in column 'age', 0 if 'male').
    - CourseLevel_Upper: 1 if 'students' == 'upper', 0 if 'lower'.
    - NativeSpeaker: indicator mapped from 'allstudents' (yes=1/no=0).
    - TenureTrack: indicator mapped from 'eval' (yes=1/no=0).
    - Enrolled: numeric from 'credits' (number enrolled).
    - LogEnrolled: log(Enrolled) (for modeling).
    - NumRespondents: numeric from 'minority' (number of respondents).
    - Age: numeric instructor age (from 'division').
    - InstructorID: numeric instructor identifier (from 'rownames').

    Notes: The original dataset contains a number of misaligned column descriptions; the mapping above follows the classical 'beauty' dataset variable layout used in the literature (prof = continuous beauty rating; tenure = evaluation score; etc.). Rows with missing EvalScore or BeautyScore are dropped.
    """
    df = df.copy()

    # Map and coerce columns to the analysis names
    # Dependent variable: overall evaluation score (in this dataset stored in 'tenure')
    df['EvalScore'] = pd.to_numeric(df.get('tenure'), errors='coerce')

    # Primary IV: continuous beauty rating (panel average) stored in 'prof'
    df['BeautyScore'] = pd.to_numeric(df.get('prof'), errors='coerce')

    # Optional binary beauty flag if available (some versions have 'beauty' factor yes/no)
    if 'beauty' in df.columns:
        df['BeautyBinary'] = df['beauty'].map({"yes": 1, "no": 0})

    # Gender mapping: column named 'age' actually contains 'male'/'female' in this schema
    if 'age' in df.columns:
        df['Female'] = df['age'].map({"female": 1, "male": 0})

    # Course level: 'students' = 'upper'/'lower'
    if 'students' in df.columns:
        df['CourseLevel_Upper'] = df['students'].map({"upper": 1, "lower": 0})

    # Native English speaker indicator: mapped from 'allstudents' yes/no (schema note mismatch)
    if 'allstudents' in df.columns:
        df['NativeSpeaker'] = df['allstudents'].map({"yes": 1, "no": 0})

    # Tenure-track indicator: mapped from 'eval' yes/no per the provided schema descriptions
    if 'eval' in df.columns:
        df['TenureTrack'] = df['eval'].map({"yes": 1, "no": 0})

    # Enrollment and number of respondents
    df['Enrolled'] = pd.to_numeric(df.get('credits'), errors='coerce')
    df['NumRespondents'] = pd.to_numeric(df.get('minority'), errors='coerce')

    # Instructor age (from 'division' per schema mapping)
    df['Age'] = pd.to_numeric(df.get('division'), errors='coerce')

    # Instructor identifier for clustering standard errors (from 'rownames')
    df['InstructorID'] = pd.to_numeric(df.get('rownames'), errors='coerce')

    # Drop rows missing the core DV or IV
    df = df.dropna(subset=['EvalScore', 'BeautyScore'])

    # Create centered and transformed variables used in the model
    df['BeautyScore_c'] = df['BeautyScore'] - df['BeautyScore'].mean()

    # Log-transform enrollment for scale stability (clip to avoid log(0))
    df['LogEnrolled'] = np.log(df['Enrolled'].clip(lower=1))

    # Ensure all required columns exist (fill with NaN if absent so downstream code has consistent columns)
    required = [
        'EvalScore', 'BeautyScore', 'BeautyScore_c', 'BeautyBinary', 'Female',
        'CourseLevel_Upper', 'NativeSpeaker', 'TenureTrack', 'Enrolled', 'LogEnrolled',
        'NumRespondents', 'Age', 'InstructorID'
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Return only the columns necessary for modeling (keeps order predictable)
    return df[required]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Estimate the effect of instructor beauty on student evaluations.

    Model: OLS with cluster-robust standard errors at the instructor level.
    Formula: EvalScore ~ BeautyScore_c + controls
    Returns the fitted model object (statsmodels regression results).
    """
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Define the regression formula
    formula = (
        'EvalScore ~ BeautyScore_c + Female + CourseLevel_Upper + NativeSpeaker '
        '+ TenureTrack + LogEnrolled + Age'
    )

    # Fit OLS and compute cluster-robust SEs by InstructorID
    model_fit = smf.ols(formula=formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['InstructorID']}
    )

    # Print a concise summary and return the fitted results object
    print(model_fit.summary())

    # Also return a small dictionary with the key coefficient and standard error for convenience
    results = {
        'model': model_fit,
        'coef_beauty': model_fit.params.get('BeautyScore_c'),
        'se_beauty_clustered': model_fit.bse.get('BeautyScore_c')
    }
    return results


