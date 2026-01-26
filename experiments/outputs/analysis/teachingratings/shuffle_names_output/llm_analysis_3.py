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
    Transform the raw dataset into a clean dataframe for modeling.

    Produces the following columns used in the model:
      - TeachingEval: dependent variable (from 'tenure')
      - Beauty_c: centered beauty score (from 'prof')
      - Female: binary indicator (from 'age')
      - InstructorAge: numeric instructor age (from 'division')
      - LogClassSize: log(1 + number of respondents) (from 'minority')
      - UpperDivision: binary indicator from 'students'
      - NativeEnglish: binary indicator from 'allstudents'
      - TenureTrack: binary indicator from 'eval'

    The function drops rows missing the DV or the main IV and constructs all control columns.
    """
    # Work on a copy
    df = df.copy()

    # Rename and create core variables
    # Teaching evaluation (DV) -- according to the provided schema this is in column 'tenure'
    df['TeachingEval'] = pd.to_numeric(df['tenure'], errors='coerce')

    # Beauty score (IV) -- numeric 'prof' is the average beauty rating (mean=0 shifted in original)
    df['BeautyScore'] = pd.to_numeric(df['prof'], errors='coerce')

    # Drop rows missing DV or IV
    df = df.dropna(subset=['TeachingEval', 'BeautyScore'])

    # Create centered beauty variable for interpretability
    df['Beauty_c'] = df['BeautyScore'] - df['BeautyScore'].mean()

    # Controls
    # Instructor gender: dataset's 'age' column encodes gender per the schema ('male'/'female')
    df['age_str'] = df['age'].astype(str).str.lower()
    df['Female'] = df['age_str'].map(lambda x: 1 if x == 'female' else 0)

    # Instructor chronological age (numeric) from 'division'
    df['InstructorAge'] = pd.to_numeric(df['division'], errors='coerce')

    # Class size / number of raters: original column 'minority' stores the number of students who participated
    df['ClassSize'] = pd.to_numeric(df['minority'], errors='coerce')
    # Use log transform to reduce skew (add 1 to avoid log(0))
    df['LogClassSize'] = np.log(df['ClassSize'].fillna(0) + 1)

    # Course level: upper/lower division from 'students' column
    df['students_str'] = df['students'].astype(str).str.lower()
    df['UpperDivision'] = df['students_str'].map(lambda x: 1 if x == 'upper' else 0)

    # Native English speaker indicator from 'allstudents' (yes/no per schema)
    df['allstudents_str'] = df['allstudents'].astype(str).str.lower()
    df['NativeEnglish'] = df['allstudents_str'].map(lambda x: 1 if x == 'yes' else 0)

    # Tenure track indicator from 'eval' (yes/no per schema)
    df['eval_str'] = df['eval'].astype(str).str.lower()
    df['TenureTrack'] = df['eval_str'].map(lambda x: 1 if x == 'yes' else 0)

    # Final selection: keep only the columns required for modeling
    keep_cols = [
        'TeachingEval',
        'BeautyScore',
        'Beauty_c',
        'Female',
        'InstructorAge',
        'ClassSize',
        'LogClassSize',
        'UpperDivision',
        'NativeEnglish',
        'TenureTrack'
    ]

    # Ensure numeric columns are numeric
    for c in ['BeautyScore', 'Beauty_c', 'InstructorAge', 'ClassSize', 'LogClassSize', 'TeachingEval']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Return only rows that have no missing values in the final model columns (conservative approach)
    df = df[keep_cols].dropna()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model predicting TeachingEval from centered beauty and controls.

    Model specification:
      TeachingEval ~ Beauty_c + Female + InstructorAge + LogClassSize + UpperDivision + NativeEnglish + TenureTrack

    Returns the statsmodels results object with HC3 robust standard errors.
    """
    # Prepare X and y
    y = df['TeachingEval']

    X = df[['Beauty_c', 'Female', 'InstructorAge', 'LogClassSize', 'UpperDivision', 'NativeEnglish', 'TenureTrack']].copy()
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS
    ols_res = sm.OLS(y, X).fit()

    # Get robust covariance (HC3) for heteroskedasticity-robust inference
    robust_res = ols_res.get_robustcov_results(cov_type='HC3')

    # Print a concise summary (users can further inspect robust_res)
    print(robust_res.summary())

    return robust_res


