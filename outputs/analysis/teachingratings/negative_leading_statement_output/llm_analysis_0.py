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
    """
    Transform the raw Hamermesh dataset into an analysis-ready dataframe.

    Steps:
    - Copy the dataframe to avoid mutating the input.
    - Drop rows missing the outcome ('eval') or the key predictor ('beauty').
    - Ensure numeric columns are numeric; drop rows with missing essential numeric values.
    - Create binary indicator (0/1) columns for categorical controls with explicit names used in the model.
    - Create log-transformed class-size variables to reduce skew.
    - Mean-center the beauty variable to improve interpretability of the intercept.

    The returned dataframe contains the exact column names referenced in the conceptual variables and modeling code.
    """
    df = df.copy()

    # Drop rows missing outcome or main predictor
    df = df.dropna(subset=['eval', 'beauty'])

    # Ensure core numeric columns are numeric; coerce errors -> NaN
    numeric_cols = ['age', 'students', 'allstudents', 'prof']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing essential numeric controls (students, allstudents, age, prof) because they are needed for controls/clustering
    df = df.dropna(subset=['students', 'allstudents', 'age', 'prof'])

    # Create explicit binary/dummy control columns with exact names used in modeling
    # Use defensive checks in case some levels are missing
    df['gender_female'] = (df['gender'] == 'female').astype(int) if 'gender' in df.columns else 0
    df['minority_yes'] = (df['minority'] == 'yes').astype(int) if 'minority' in df.columns else 0
    df['credits_single'] = (df['credits'] == 'single').astype(int) if 'credits' in df.columns else 0
    df['division_lower'] = (df['division'] == 'lower').astype(int) if 'division' in df.columns else 0
    df['native_yes'] = (df['native'] == 'yes').astype(int) if 'native' in df.columns else 0
    df['tenure_yes'] = (df['tenure'] == 'yes').astype(int) if 'tenure' in df.columns else 0

    # Log-transform class-size variables to reduce skew (add a small constant if zeros appear, though dataset minimums suggest >0)
    df['log_students'] = np.log(df['students'].astype(float) + 1e-6)
    df['log_allstudents'] = np.log(df['allstudents'].astype(float) + 1e-6)

    # Mean-center beauty for interpretability
    df['beauty_c'] = df['beauty'].astype(float) - float(df['beauty'].astype(float).mean())

    # Final verification: keep only rows without missing values in columns that will be used by the model
    required_cols = ['eval', 'beauty_c', 'age', 'gender_female', 'minority_yes', 'credits_single',
                     'division_lower', 'native_yes', 'tenure_yes', 'log_students', 'log_allstudents', 'prof']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run a set of OLS models to test whether instructor beauty predicts student evaluations.

    Returns a dictionary with two statsmodels regression results objects:
    - 'baseline': OLS with only beauty (mean-centered) as predictor (clustered SE by prof)
    - 'adjusted': OLS with beauty + controls listed in the conceptual variables (clustered SE by prof)

    Clustered standard errors by 'prof' account for non-independence of multiple courses taught by the same instructor.
    """
    # Ensure we operate on a dataframe that contains the transformed columns
    df = df.copy()

    # Baseline model: eval ~ beauty
    X_base = sm.add_constant(df[['beauty_c']])
    y = df['eval']
    model_base = sm.OLS(y, X_base)
    results_base = model_base.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Adjusted model: add demographic and course controls
    control_cols = ['age', 'gender_female', 'minority_yes', 'credits_single', 'division_lower',
                    'native_yes', 'tenure_yes', 'log_students', 'log_allstudents']
    # Keep only controls that exist in the dataframe (defensive)
    control_cols = [c for c in control_cols if c in df.columns]
    X_adj = sm.add_constant(df[['beauty_c'] + control_cols])
    model_adj = sm.OLS(y, X_adj)
    results_adj = model_adj.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Useful summary outputs can be printed by the caller; return raw results for programmatic inspection
    return {
        'baseline': results_base,
        'adjusted': results_adj
    }

# Example usage (not executed here):
# df_trans = transform(raw_df)
# res = model(df_trans)
# print(res['baseline'].summary())
# print(res['adjusted'].summary())


