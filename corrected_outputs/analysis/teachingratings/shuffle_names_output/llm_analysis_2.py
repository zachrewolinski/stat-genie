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
    Transform the raw Hamermesh classroom dataset into analytical variables.

    Produces the following columns used in modeling:
      - EvalScore: numeric teaching evaluation score (DV)
      - BeautyScore: panel average attractiveness rating (IV, continuous)
      - BeautyScore_z: standardized BeautyScore (z-score)
      - BeautyHigh: binary attractiveness indicator when a 'beauty' yes/no column exists (1=yes)
      - IsFemale, Age, TenureTrack, NativeEnglish, UpperDiv, ClassSize, Enrollment, IsMinority
      - InstructorID: numeric identifier for clustering

    The dataset schema provided includes some label mismatches (e.g., 'tenure' described as evaluation score,
    'prof' described as the attractiveness rating). This function attempts to map columns according to those
    descriptions. Inspect the transformed dataframe after running to ensure columns align with your local copy.
    """
    df = df.copy()

    # Coerce core numeric columns using the schema mapping assumptions
    # 'prof' in the schema is the averaged attractiveness rating (continuous)
    if 'prof' in df.columns:
        df['BeautyScore'] = pd.to_numeric(df['prof'], errors='coerce')
    else:
        # If 'prof' missing, attempt to create from other columns or set NaN
        df['BeautyScore'] = np.nan

    # 'tenure' column is documented in the schema as the course evaluation score
    if 'tenure' in df.columns:
        df['EvalScore'] = pd.to_numeric(df['tenure'], errors='coerce')
    else:
        # fallback: try 'eval' if it contains numeric values
        df['EvalScore'] = pd.to_numeric(df.get('eval', np.nan), errors='coerce')

    # Binary 'BeautyHigh' derived from 'beauty' column if present (yes/no), otherwise median split
    if 'beauty' in df.columns:
        df['BeautyHigh'] = df['beauty'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    else:
        df['BeautyHigh'] = np.nan

    # Gender: schema indicates the column 'age' contains 'male'/'female' labels in this dataset
    if 'age' in df.columns:
        df['IsFemale'] = df['age'].astype(str).str.lower().map({'female': 1, 'male': 0})
    else:
        # try the 'gender' column if present
        if 'gender' in df.columns:
            df['IsFemale'] = df['gender'].astype(str).str.lower().map({'female': 1, 'male': 0})
        else:
            df['IsFemale'] = np.nan

    # Age numeric mapped from 'division' per the provided schema notes
    if 'division' in df.columns:
        df['Age'] = pd.to_numeric(df['division'], errors='coerce')
    else:
        df['Age'] = np.nan

    # Tenure-track indicator mapped from 'eval' column (schema ambiguous: 'eval' described as tenure-track yes/no)
    if 'eval' in df.columns:
        df['TenureTrack'] = df['eval'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    else:
        df['TenureTrack'] = np.nan

    # Native English speaker indicator from 'allstudents' per schema
    if 'allstudents' in df.columns:
        df['NativeEnglish'] = df['allstudents'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    else:
        df['NativeEnglish'] = np.nan

    # Course level (upper/lower)
    if 'students' in df.columns:
        df['UpperDiv'] = df['students'].astype(str).str.lower().map({'upper': 1, 'lower': 0})
    else:
        df['UpperDiv'] = np.nan

    # Class size and enrollment proxies
    if 'minority' in df.columns:
        # schema indicates 'minority' stores number of student respondents in some descriptions
        df['ClassSize'] = pd.to_numeric(df['minority'], errors='coerce')
    else:
        df['ClassSize'] = np.nan

    if 'credits' in df.columns:
        df['Enrollment'] = pd.to_numeric(df['credits'], errors='coerce')
    else:
        df['Enrollment'] = np.nan

    # IsMinority: some schema descriptions suggest 'beauty' column actually stores minority status
    if 'beauty' in df.columns:
        df['IsMinority'] = df['beauty'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    else:
        df['IsMinority'] = np.nan

    # Instructor ID for clustering (rownames column corresponds to instructor identifier in provided schema)
    if 'rownames' in df.columns:
        df['InstructorID'] = pd.to_numeric(df['rownames'], errors='coerce')
    else:
        # fall back to an index-based id
        df['InstructorID'] = np.arange(len(df))

    # Final cleaning: drop rows without DV or IV
    df = df.dropna(subset=['EvalScore', 'BeautyScore'])

    # Standardize BeautyScore to a z-score for interpretability
    if df['BeautyScore'].std(ddof=0) == 0 or np.isnan(df['BeautyScore'].std(ddof=0)):
        df['BeautyScore_z'] = df['BeautyScore']
    else:
        df['BeautyScore_z'] = (df['BeautyScore'] - df['BeautyScore'].mean()) / df['BeautyScore'].std(ddof=0)

    # If BeautyHigh was not available, create a median split on BeautyScore
    if 'BeautyHigh' in df.columns and df['BeautyHigh'].isnull().all():
        df['BeautyHigh'] = (df['BeautyScore'] > df['BeautyScore'].median()).astype(int)

    # Return transformed dataframe containing all required columns
    keep_cols = [
        'EvalScore', 'BeautyScore', 'BeautyScore_z', 'BeautyHigh',
        'IsFemale', 'Age', 'TenureTrack', 'NativeEnglish', 'UpperDiv',
        'ClassSize', 'Enrollment', 'IsMinority', 'InstructorID'
    ]
    # ensure columns exist even if NaN
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols + [c for c in df.columns if c not in keep_cols]]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating the effect of instructor beauty on teaching evaluations.

    Primary model: EvalScore ~ BeautyScore_z + controls
    Standard errors are clustered by InstructorID when possible. If clustering fails, falls back to HC3 robust SE.

    Returns:
      - statsmodels regression results object.
    """
    df = df.copy()

    # Define dependent and independent variables
    y = df['EvalScore']

    # Controls included to reduce omitted variable bias: gender, age, tenure-track, native English,
    # course level, class size, enrollment, minority indicator
    X_cols = [
        'BeautyScore_z',
        'IsFemale',
        'Age',
        'TenureTrack',
        'NativeEnglish',
        'UpperDiv',
        'ClassSize',
        'Enrollment',
        'IsMinority'
    ]

    # Build design matrix and drop observations with missing covariates
    X = df[X_cols].astype(float)
    X = sm.add_constant(X)

    valid = X.notnull().all(axis=1) & y.notnull()
    y_clean = y[valid]
    X_clean = X[valid]
    groups = df.loc[valid, 'InstructorID'] if 'InstructorID' in df.columns else None

    ols_mod = sm.OLS(y_clean, X_clean)

    # Try clustered SE by InstructorID (preferred). If InstructorID missing or fit fails, use HC3 robust SE.
    try:
        if groups is None or groups.isnull().all():
            raise ValueError('No valid InstructorID for clustering; using HC3 instead')
        results = ols_mod.fit(cov_type='cluster', cov_kwds={'groups': groups})
    except Exception:
        results = ols_mod.fit(cov_type='HC3')

    # Print summary for quick inspection
    print(results.summary())

    return results


