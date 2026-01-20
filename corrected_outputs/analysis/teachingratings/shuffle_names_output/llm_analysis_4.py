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
    Transform the raw Hamermesh dataset into a dataframe ready for modeling.

    Inputs:
    - df: original dataframe with columns as provided in the schema.

    Outputs (columns preserved/created):
    - EvalScore: numeric teaching evaluation score (from 'tenure')
    - BeautyScore: original continuous beauty rating (from 'prof')
    - Beauty_z: standardized (z) beauty score
    - BeautyBinary: binary indicator from 'beauty' column (yes -> 1, else 0)
    - Female: indicator (1=female) from 'age' column (which contains gender labels in this schema)
    - Age: numeric instructor age from 'division'
    - ClassSize: numeric from 'credits'
    - NumResponses: numeric from 'minority'
    - LowerDivision: indicator from 'students' ('lower' -> 1)
    - NativeEnglish: indicator from 'allstudents' ('yes' -> 1)
    - TenureTrack: indicator from 'eval' ('yes' -> 1)
    - ProfessorID: professor identifier from 'rownames' (kept for clustering)
    """
    df = df.copy()

    # Ensure we operate on expected columns; coerce types conservatively
    # Dependent and main independent variable
    df['EvalScore'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['BeautyScore'] = pd.to_numeric(df['prof'], errors='coerce')

    # Basic drop for rows missing DV or main IV
    df = df.dropna(subset=['EvalScore', 'BeautyScore'])

    # Binary beauty indicator (robust to capitalization and minor variants)
    def map_yes_no(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        return 1 if s in ('yes', 'y', '1', 'true') else 0

    if 'beauty' in df.columns:
        df['BeautyBinary'] = df['beauty'].apply(map_yes_no).astype(int)
    else:
        df['BeautyBinary'] = 0

    # Gender: in this schema the 'age' column contains gender labels ('male'/'female')
    if 'age' in df.columns:
        df['Female'] = df['age'].astype(str).str.strip().str.lower().apply(lambda x: 1 if x == 'female' else 0)
    else:
        df['Female'] = 0

    # Age numeric (from 'division' in this schema)
    df['Age'] = pd.to_numeric(df['division'], errors='coerce')

    # Class size / enrolled students (from 'credits' in this schema)
    df['ClassSize'] = pd.to_numeric(df['credits'], errors='coerce')

    # Number of student responses (from 'minority' in this schema)
    df['NumResponses'] = pd.to_numeric(df['minority'], errors='coerce')

    # Lower-division indicator from 'students' ('lower'/'upper')
    if 'students' in df.columns:
        df['LowerDivision'] = df['students'].astype(str).str.strip().str.lower().apply(lambda x: 1 if 'lower' in x else 0)
    else:
        df['LowerDivision'] = 0

    # Native English speaker indicator from 'allstudents' (yes/no in schema)
    if 'allstudents' in df.columns:
        df['NativeEnglish'] = df['allstudents'].astype(str).str.strip().str.lower().apply(lambda x: 1 if x in ('yes','y','true','1') else 0)
    else:
        df['NativeEnglish'] = 0

    # Tenure-track indicator from 'eval' (yes/no in this schema)
    if 'eval' in df.columns:
        df['TenureTrack'] = df['eval'].astype(str).str.strip().str.lower().apply(lambda x: 1 if x in ('yes','y','true','1') else 0)
    else:
        df['TenureTrack'] = 0

    # Professor identifier used for clustering
    if 'rownames' in df.columns:
        df['ProfessorID'] = df['rownames']
    else:
        # fallback: if native exists and is unique, use that
        df['ProfessorID'] = df.get('native', pd.Series(np.arange(len(df)), index=df.index))

    # Standardize beauty score (z-score) for interpretability
    df['Beauty_z'] = (df['BeautyScore'] - df['BeautyScore'].mean()) / (df['BeautyScore'].std(ddof=0) if df['BeautyScore'].std(ddof=0) != 0 else 1)

    # Final conservative drop: remove rows with missing values in core controls to ensure consistent sample
    required_for_model = ['EvalScore', 'Beauty_z', 'Female', 'Age', 'ClassSize', 'NumResponses', 'LowerDivision', 'NativeEnglish', 'TenureTrack', 'ProfessorID']
    df = df.dropna(subset=required_for_model)

    # Keep only columns that are necessary/created for modeling (plus original identifiers if desired)
    keep_cols = required_for_model + ['BeautyScore', 'BeautyBinary']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating the effect of beauty on teaching evaluations.

    Model:
      EvalScore ~ Beauty_z + Female + Age + ClassSize + NumResponses + LowerDivision + NativeEnglish + TenureTrack

    We cluster standard errors by ProfessorID to account for within-instructor correlation when a professor appears multiple times.

    Returns the fitted statsmodels regression results object.
    """
    import statsmodels.formula.api as smf

    formula = (
        'EvalScore ~ Beauty_z + Female + Age + ClassSize + NumResponses + '
        'LowerDivision + NativeEnglish + TenureTrack'
    )

    # Fit OLS and cluster standard errors by ProfessorID
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['ProfessorID']})

    return model


