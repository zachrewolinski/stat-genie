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
    Transform the raw Hamermesh & Parker classroom dataset for analysis.

    Steps:
    - Drop rows missing the main variables (beauty, eval).
    - Create binary indicators for categorical control variables.
    - Standardize beauty into beauty_z for easier coefficient interpretation.
    - Log-transform class-size variables (students, allstudents) to reduce skew.
    - Convert prof to a categorical variable (used for fixed effects / clustering).
    - Drop rows with missing values in any of the model variables.

    Returns a dataframe that contains all columns referenced in the modeling function.
    """
    df = df.copy()

    # Ensure expected columns exist
    required_cols = ['beauty', 'eval', 'age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students', 'allstudents', 'prof']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows missing primary variables
    df = df.dropna(subset=['beauty', 'eval'])

    # Normalize string columns to lower-case (safe mapping)
    for col in ['gender', 'minority', 'credits', 'division', 'native', 'tenure']:
        # If column is not object type, convert to string first to avoid errors
        if col in df.columns and df[col].dtype != 'O':
            df[col] = df[col].astype(str)
        df[col] = df[col].str.strip().str.lower()

    # Binary indicators (map common choices; unexpected values will become NaN)
    df['is_female'] = df['gender'].map({'female': 1, 'male': 0})
    df['is_minority'] = df['minority'].map({'yes': 1, 'no': 0})
    df['is_single_credit'] = df['credits'].map({'single': 1, 'more': 0})
    df['is_upper'] = df['division'].map({'upper': 1, 'lower': 0})
    df['is_native'] = df['native'].map({'yes': 1, 'no': 0})
    df['is_tenure'] = df['tenure'].map({'yes': 1, 'no': 0})

    # Standardize beauty (z-score)
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1.0
    df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Log-transform student counts to reduce skew. Use np.log for positive counts.
    df['ln_students'] = np.log(df['students'].astype(float))
    df['ln_allstudents'] = np.log(df['allstudents'].astype(float))

    # Convert prof to categorical (keeps original numeric codes but marks as category)
    df['prof'] = df['prof'].astype('category')

    # Drop rows with missing values in any of the variables that will be used in the model
    model_cols = [
        'beauty_z', 'eval', 'age', 'is_female', 'is_minority', 'is_single_credit',
        'is_upper', 'is_native', 'is_tenure', 'ln_students', 'ln_allstudents', 'prof'
    ]
    df = df.dropna(subset=model_cols)

    # Cast control indicator columns to numeric (int)
    for c in ['is_female', 'is_minority', 'is_single_credit', 'is_upper', 'is_native', 'is_tenure']:
        df[c] = df[c].astype(int)

    # Ensure age numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])

    # Final check: keep only columns necessary for modeling plus original identifiers
    keep_cols = list(set(model_cols + ['beauty', 'students', 'allstudents', 'gender', 'minority', 'credits', 'division', 'native', 'tenure']))
    # Preserve original column order where possible
    keep_cols_sorted = [c for c in df.columns if c in keep_cols]
    df = df[keep_cols_sorted]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS models to estimate the association between instructor beauty and student evaluations.

    Two primary specifications are estimated:
    1) Baseline OLS with observed controls and heteroskedasticity-robust (HC3) standard errors.
    2) OLS including professor fixed effects (C(prof)) with standard errors clustered by professor.

    Returns a dictionary with fitted result objects for both specifications.
    """
    # Local import for formula API
    import statsmodels.formula.api as smf

    # Ensure expected transformed columns exist
    required = ['eval', 'beauty_z', 'age', 'is_female', 'is_minority', 'is_single_credit',
                'is_upper', 'is_native', 'is_tenure', 'ln_students', 'ln_allstudents', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing columns required for modeling: {missing}")

    # Define the common RHS of the formula (controls)
    controls = 'age + is_female + is_minority + is_single_credit + is_upper + is_native + is_tenure + ln_students + ln_allstudents'

    # Baseline OLS (no professor fixed effects). Use robust (HC3) standard errors to be conservative.
    formula_base = f'eval ~ beauty_z + {controls}'
    model_base = smf.ols(formula_base, data=df).fit(cov_type='HC3')

    # OLS with professor fixed effects. Cluster standard errors by professor (prof) to account for within-prof correlation.
    # Using C(prof) adds a dummy for each professor.
    formula_fe = f'eval ~ beauty_z + {controls} + C(prof)'
    # For clustering, pass groups via cov_kwds
    model_fe = smf.ols(formula_fe, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Package results. Caller can access e.g. results['baseline'].summary()
    results = {
        'baseline_HC3': model_base,
        'fe_with_clustered_se': model_fe
    }

    return results


