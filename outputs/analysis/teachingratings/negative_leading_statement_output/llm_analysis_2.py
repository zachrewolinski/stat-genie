from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/negative_leading_statement_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure expected columns exist
    expected = ['beauty','eval','minority','age','gender','credits','division','native','tenure','students','prof']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Drop rows missing the key variables (DV or IV) or essential controls
    df = df.dropna(subset=['eval','beauty','students','prof'])

    # Convert columns to appropriate dtypes
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['age','students','prof'])

    # Create mean-centered beauty and quadratic term to capture nonlinearity
    df['beauty_c'] = df['beauty'] - df['beauty'].mean()
    df['beauty_sq'] = df['beauty_c'] ** 2

    # Binary indicators for categorical controls (consistent naming used in model)
    df['minority_binary'] = df['minority'].map({'yes': 1, 'no': 0})
    df['gender_male'] = (df['gender'].astype(str).str.lower() == 'male').astype(int)
    df['credits_single'] = (df['credits'].astype(str).str.lower() == 'single').astype(int)
    df['division_upper'] = (df['division'].astype(str).str.lower() == 'upper').astype(int)
    df['native_yes'] = df['native'].map({'yes': 1, 'no': 0})
    df['tenure_yes'] = df['tenure'].map({'yes': 1, 'no': 0})

    # Log transform of students (add small constant if there are zeros, but dataset min is 5)
    df['log_students'] = np.log(df['students'].astype(float))

    # Keep only rows that have all newly created control columns non-missing
    required_cols = ['beauty_c','beauty_sq','minority_binary','age','gender_male',
                     'credits_single','division_upper','native_yes','tenure_yes','log_students','prof','eval']
    df = df.dropna(subset=required_cols)

    # Return transformed dataframe; do not drop original columns so user can inspect raw data
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Define regressors and outcome exactly as in the conceptual variables
    X_cols = [
        'beauty_c', 'beauty_sq',
        'minority_binary', 'age', 'gender_male',
        'credits_single', 'division_upper', 'native_yes', 'tenure_yes',
        'log_students'
    ]

    # Ensure columns exist
    missing = [c for c in X_cols + ['eval','prof'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    X = df[X_cols].astype(float)
    X = sm.add_constant(X)
    y = df['eval'].astype(float)

    # Fit OLS
    ols_res = sm.OLS(y, X).fit()

    # Obtain cluster-robust standard errors clustered by professor id (prof)
    try:
        clustered_res = ols_res.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        # Fallback: heteroskedasticity-robust (HC3) if clustering fails
        clustered_res = ols_res.get_robustcov_results(cov_type='HC3')

    # Print summary (helpful for interactive use); return the robust-results object
    print(clustered_res.summary())

    # For programmatic use, also return a compact dictionary of key inference quantities
    out = {
        'results': clustered_res,
        'coef': clustered_res.params,
        'pvalues': clustered_res.pvalues,
        'conf_int': clustered_res.conf_int()
    }
    return out


