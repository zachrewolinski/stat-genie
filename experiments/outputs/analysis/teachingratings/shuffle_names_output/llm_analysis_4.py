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
    Transform raw classroom/teacher dataset into analysis-ready dataframe.

    Produces the following columns (kept in the returned dataframe):
    - EvalScore: numeric course evaluation score (from 'tenure' column per dataset schema)
    - BeautyScore: raw numeric beauty rating (from 'prof')
    - Beauty_z: standardized beauty score (z-score)
    - Beauty_z2: squared z-score to capture non-linearity
    - gender_male: binary indicator for male instructor (1 = male, 0 = female/other)
    - course_upper: binary indicator for upper-division course (1 = upper, 0 = lower/other)
    - native_english: binary indicator if instructor is native English speaker (1 = yes)
    - tenure_track: binary indicator if instructor is on tenure track (1 = yes)
    - class_size_log: log(1 + credits) where 'credits' used as class enrollment
    - response_rate: NumRespondents / ClassSize (uses 'minority' as NumRespondents per schema)
    - instructor_id: identifier from 'rownames' (if present) used for clustering
    """
    df = df.copy()

    # Basic required columns: beauty rating and evaluation score
    # 'prof' (beauty numeric) and 'tenure' (evaluation score) per provided schema
    df = df.dropna(subset=['prof', 'tenure'])

    # Primary variables
    df['EvalScore'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['BeautyScore'] = pd.to_numeric(df['prof'], errors='coerce')

    # Standardize beauty (z-score) and add quadratic term
    beauty_mean = df['BeautyScore'].mean()
    beauty_std = df['BeautyScore'].std(ddof=0)
    if pd.isna(beauty_std) or beauty_std == 0:
        # fallback to avoid division by zero
        df['Beauty_z'] = 0.0
    else:
        df['Beauty_z'] = (df['BeautyScore'] - beauty_mean) / beauty_std
    df['Beauty_z2'] = df['Beauty_z'] ** 2

    # Controls: robust parsing of categorical columns (they are labelled inconsistently in schema)
    # gender: stored in column 'age' per provided schema (values like 'male'/'female')
    df['gender_male'] = df['age'].astype(str).str.lower().str.contains('male', na=False).astype(int)

    # course level: 'students' column containing 'upper'/'lower'
    df['course_upper'] = df['students'].astype(str).str.lower().str.contains('upper', na=False).astype(int)

    # native English speaker: 'allstudents' column coded yes/no per schema
    df['native_english'] = df['allstudents'].astype(str).str.lower().isin(['yes', 'y', 'true', '1']).astype(int)

    # tenure track indicator: 'eval' column coded yes/no per schema
    df['tenure_track'] = df['eval'].astype(str).str.lower().isin(['yes', 'y', 'true', '1']).astype(int)

    # Class size and respondents: use 'credits' as class enrollment and 'minority' as number of respondents
    df['ClassSize'] = pd.to_numeric(df['credits'], errors='coerce')
    df['NumRespondents'] = pd.to_numeric(df['minority'], errors='coerce')

    # log transform class size to reduce skew
    df['class_size_log'] = np.log1p(df['ClassSize'].fillna(0))

    # response rate: safe division, clip to [0,1]
    df['response_rate'] = np.where(
        (df['ClassSize'] > 0) & (~df['NumRespondents'].isna()),
        df['NumRespondents'] / df['ClassSize'],
        np.nan
    )
    # Clip extreme values
    df['response_rate'] = df['response_rate'].clip(lower=0, upper=1)

    # Instructor identifier for clustering (if present)
    if 'rownames' in df.columns:
        # rownames appears to be an instructor identifier per schema
        df['instructor_id'] = df['rownames']
    else:
        df['instructor_id'] = np.nan

    # Keep only rows with a numeric EvalScore
    df = df.dropna(subset=['EvalScore'])

    # Final columns used in modeling are retained; return entire df with added columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit OLS regression of EvalScore on (standardized) Beauty and controls.

    Specification:
      EvalScore ~ Beauty_z + Beauty_z2 + gender_male + course_upper + native_english
                 + tenure_track + class_size_log + response_rate

    Standard errors are clustered by instructor_id when available.
    """
    # Columns used in the model
    X_cols = [
        'Beauty_z',
        'Beauty_z2',
        'gender_male',
        'course_upper',
        'native_english',
        'tenure_track',
        'class_size_log',
        'response_rate'
    ]

    # Drop rows with missing predictors or outcome
    model_df = df.dropna(subset=['EvalScore'] + X_cols).copy()

    if model_df.shape[0] == 0:
        raise ValueError('No observations remain after dropping missing values for the model.')

    # Design matrix
    X = model_df[X_cols]
    X = sm.add_constant(X)
    y = model_df['EvalScore']

    # Fit OLS with clustering by instructor_id when instructor_id exists and has >1 cluster
    use_cluster = ('instructor_id' in model_df.columns) and (model_df['instructor_id'].nunique() > 1)
    if use_cluster:
        # Ensure instructor_id has no missing values for clustering
        cluster_groups = model_df['instructor_id']
        # Fit with clustered standard errors
        results = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': cluster_groups})
    else:
        results = sm.OLS(y, X).fit(cov_type='HC3')

    # Print a concise summary to the console (useful when running interactively)
    try:
        print(results.summary())
    except Exception:
        pass

    return results


