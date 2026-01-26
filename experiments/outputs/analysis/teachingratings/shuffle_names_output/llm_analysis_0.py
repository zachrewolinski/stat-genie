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
    Transform the raw Hamermesh classroom dataset into a modeling dataframe.

    Creates/normalizes the following columns used by the model:
      - EvalScore: numeric course evaluation score (from 'tenure' column in the provided schema)
      - BeautyIndex: continuous beauty index (from 'prof')
      - BeautyBinary: binary beauty coder variable from 'beauty' ('yes'->1, 'no'->0)
      - Male: indicator from 'age' column mapping 'male'->1, 'female'->0
      - UpperDivision: from 'students' mapping 'upper'->1, 'lower'->0
      - Enrollment: numeric from 'credits'
      - NumRespondents: numeric from 'minority' (number of respondents in the schema)
      - NativeEnglish: from 'allstudents' ('yes'->1, 'no'->0)
      - TenureTrack: from 'eval' ('yes'->1, 'no'->0)

    Notes: The dataset schema provided has some column-description misalignments relative to typical Hamermesh data; this transform assumes the numeric columns and factor codings per the schema fields and tries to coerce/clean robustly.
    """
    df = df.copy()

    # Dependent variable: course evaluation score. Per schema this appears in 'tenure' (numeric ~2.1-5.0)
    df['EvalScore'] = pd.to_numeric(df.get('tenure'), errors='coerce')

    # Continuous beauty index: 'prof' (numeric, mean-zero beauty index in typical dataset)
    df['BeautyIndex'] = pd.to_numeric(df.get('prof'), errors='coerce')

    # Binary beauty coder: 'beauty' (map common string values yes/no -> 1/0)
    if 'beauty' in df.columns:
        df['BeautyBinary'] = df['beauty'].map({'yes': 1, 'no': 0})
        # If mapping produced NaNs (other encodings), attempt numeric coercion
        df['BeautyBinary'] = pd.to_numeric(df['BeautyBinary'], errors='coerce')
    else:
        df['BeautyBinary'] = np.nan

    # Gender: column named 'age' in schema actually contains 'male'/'female'
    if 'age' in df.columns:
        df['Male'] = df['age'].map({'male': 1, 'female': 0})
        df['Male'] = pd.to_numeric(df['Male'], errors='coerce')
    else:
        df['Male'] = np.nan

    # Course level: 'students' (lower/upper)
    if 'students' in df.columns:
        df['UpperDivision'] = df['students'].map({'upper': 1, 'lower': 0})
        df['UpperDivision'] = pd.to_numeric(df['UpperDivision'], errors='coerce')
    else:
        df['UpperDivision'] = np.nan

    # Native English indicator: 'allstudents' column in schema is 'yes'/'no'
    if 'allstudents' in df.columns:
        df['NativeEnglish'] = df['allstudents'].map({'yes': 1, 'no': 0})
        df['NativeEnglish'] = pd.to_numeric(df['NativeEnglish'], errors='coerce')
    else:
        df['NativeEnglish'] = np.nan

    # Tenure track indicator: 'eval' in schema uses yes/no for tenure-track status
    if 'eval' in df.columns:
        df['TenureTrack'] = df['eval'].map({'yes': 1, 'no': 0})
        df['TenureTrack'] = pd.to_numeric(df['TenureTrack'], errors='coerce')
    else:
        df['TenureTrack'] = np.nan

    # Enrollment and number of respondents
    df['Enrollment'] = pd.to_numeric(df.get('credits'), errors='coerce')
    df['NumRespondents'] = pd.to_numeric(df.get('minority'), errors='coerce')

    # Instructor identifier (if present) kept for possible clustering/diagnostics
    if 'rownames' in df.columns:
        df['InstructorID'] = df['rownames']

    # Drop rows missing the key variables (dependent or primary independent continuous beauty index)
    df = df.dropna(subset=['EvalScore', 'BeautyIndex'])

    # For binary/categorical controls, fill missing with 0 (conservative) and cast to int
    for c in ['BeautyBinary', 'Male', 'UpperDivision', 'NativeEnglish', 'TenureTrack']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # Keep only the columns needed for modeling (plus InstructorID for diagnostics)
    final_cols = [
        'EvalScore', 'BeautyIndex', 'BeautyBinary', 'Male', 'UpperDivision',
        'Enrollment', 'NumRespondents', 'NativeEnglish', 'TenureTrack'
    ]
    if 'InstructorID' in df.columns:
        final_cols.append('InstructorID')

    df = df[[c for c in final_cols if c in df.columns]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of EvalScore on beauty measures and controls.

    Model specification:
      EvalScore ~ BeautyIndex + BeautyBinary + Male + BeautyIndex:Male + UpperDivision + Enrollment + NumRespondents + NativeEnglish + TenureTrack

    Uses heteroskedasticity-robust (HC3) standard errors. Returns the fitted results object.
    """
    # Ensure required libraries are available in the environment
    import statsmodels.api as sm

    df = df.copy()

    # Define outcome and predictors
    y = df['EvalScore']

    # Predictor set (include interaction Male x BeautyIndex to test moderation by gender)
    predictors = [
        'BeautyIndex',
        'BeautyBinary',
        'Male',
        'UpperDivision',
        'Enrollment',
        'NumRespondents',
        'NativeEnglish',
        'TenureTrack'
    ]

    # Filter predictors that exist in df
    predictors = [p for p in predictors if p in df.columns]

    X = df[predictors].copy()

    # Add interaction term (BeautyIndex x Male) if both columns exist
    if ('BeautyIndex' in X.columns) and ('Male' in X.columns):
        X['BeautyIndex_x_Male'] = X['BeautyIndex'] * X['Male']

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust standard errors
    model = sm.OLS(y, X, missing='drop')
    results = model.fit(cov_type='HC3')

    # Print summary for quick inspection (caller can also inspect returned object)
    print(results.summary())

    return results


