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
    # Work on a copy
    df = df.copy()

    # Map/clean primary variables based on the provided schema
    # Continuous beauty score (original column 'prof')
    df['BeautyScore'] = pd.to_numeric(df['prof'], errors='coerce')

    # Binary beauty factor (original column 'beauty') -> 1/0 (yes/no)
    # Defensive mapping to handle case differences
    df['BeautyBinary'] = (
        df['beauty'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    )

    # Dependent variable: course evaluation (schema indicates 'tenure' holds eval score)
    df['EvalScore'] = pd.to_numeric(df['tenure'], errors='coerce')

    # Controls: coerce and map where necessary
    # Gender: original column 'age' in this schema actually contains 'male'/'female'
    df['Gender_Male'] = df['age'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

    # Age numeric: from 'division' per schema
    df['Age'] = pd.to_numeric(df['division'], errors='coerce')

    # Course level: 'students' contains 'upper'/'lower'
    df['UpperCourse'] = df['students'].astype(str).str.strip().str.lower().map({'upper': 1, 'lower': 0})

    # Native English speaker indicator from 'allstudents' (schema mapping)
    df['NativeEnglish'] = df['allstudents'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Tenure-track from 'eval' (schema indicates 'eval' is tenure-track flag)
    df['TenureTrack'] = df['eval'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Number of evaluation responses (interpreted from 'minority' in schema)
    df['NumEvalResponses'] = pd.to_numeric(df['minority'], errors='coerce')

    # Class size (interpreted from 'credits' in schema)
    df['ClassSize'] = pd.to_numeric(df['credits'], errors='coerce')
    df['LogClassSize'] = np.log(df['ClassSize'].clip(lower=1))

    # Professor identifier (for clustering)
    df['ProfessorID'] = df['rownames']

    # Standardize continuous beauty score (z-score) for interpretability
    df['Beauty_z'] = (df['BeautyScore'] - df['BeautyScore'].mean()) / df['BeautyScore'].std()

    # Select relevant variables and drop rows with missing DV or primary IV(s)
    required_columns = [
        'EvalScore', 'BeautyScore', 'Beauty_z', 'BeautyBinary',
        'Gender_Male', 'Age', 'UpperCourse', 'NativeEnglish', 'TenureTrack',
        'NumEvalResponses', 'ClassSize', 'LogClassSize', 'ProfessorID'
    ]

    # If these columns don't exist (because of weird input), ensure they exist in df to avoid KeyError
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows missing the DV or the primary continuous IV; we'll use listwise deletion for the model
    df = df.dropna(subset=['EvalScore', 'Beauty_z'])

    # Return transformed dataframe with all model columns included
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    # Make sure the transform step has been applied (user should call transform first)
    # Define formulas
    formula_cont = (
        'EvalScore ~ Beauty_z + Gender_Male + Age + UpperCourse + NativeEnglish + '
        'TenureTrack + NumEvalResponses + LogClassSize'
    )

    formula_bin = (
        'EvalScore ~ BeautyBinary + Gender_Male + Age + UpperCourse + NativeEnglish + '
        'TenureTrack + NumEvalResponses + LogClassSize'
    )

    # Fit OLS with cluster-robust standard errors clustered at the professor level
    # Model using continuous beauty (z-scored)
    mod1 = smf.ols(formula_cont, data=df)
    try:
        res1 = mod1.fit(cov_type='cluster', cov_kwds={'groups': df['ProfessorID']})
    except Exception:
        # Fall back to plain OLS if clustering fails
        res1 = mod1.fit()

    # Model using binary beauty indicator
    mod2 = smf.ols(formula_bin, data=df)
    try:
        res2 = mod2.fit(cov_type='cluster', cov_kwds={'groups': df['ProfessorID']})
    except Exception:
        res2 = mod2.fit()

    # Return both fitted results objects so analyst can inspect summaries, coefficients, and diagnostics
    return {
        'model_beauty_continuous': res1,
        'model_beauty_binary': res2
    }


