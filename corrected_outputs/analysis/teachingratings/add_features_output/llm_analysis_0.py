from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure key columns exist
    required = ['beauty', 'eval', 'age', 'students', 'gender', 'tenure', 'division', 'credits', 'native', 'minority', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing values in any variable we will use in the model
    df = df.dropna(subset=required)

    # Ensure numeric types
    df['beauty'] = pd.to_numeric(df['beauty'], errors='coerce')
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # After coercion, drop any new NA rows
    df = df.dropna(subset=['beauty', 'eval', 'age', 'students', 'prof'])

    # Create quadratic term for beauty to allow nonlinearity
    df['beauty_sq'] = df['beauty'] ** 2

    # Create binary female indicator (1 if female, 0 if male). Handle casing/whitespace robustly.
    df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    df['female'] = (df['gender'] == 'female').astype(int)

    # Log-transform of students to reduce skew (use natural log). students > 0 by data description; guard anyway.
    df = df[df['students'] > 0]
    df['log_students'] = np.log(df['students'].astype(float))

    # Ensure categorical controls are strings / categories (keeps original levels for Patsy/Statsmodels C() usage)
    for col in ['tenure', 'division', 'credits', 'native', 'minority']:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': pd.NA, 'None': pd.NA})
        df[col] = df[col].astype('category')

    # Ensure prof is integer (for clustering)
    df['prof'] = df['prof'].astype(int)

    # Final drop for any remaining missing values in columns used downstream
    used_cols = ['beauty', 'beauty_sq', 'eval', 'female', 'age', 'log_students', 'tenure', 'division', 'credits', 'native', 'minority', 'prof']
    df = df.dropna(subset=used_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs OLS of evaluation on beauty (linear + quadratic) with controls and an interaction of beauty by female.
    Uses clustered robust standard errors at the professor (prof) level.

    Returns a fitted results object with clustered covariance.
    """
    import statsmodels.formula.api as smf

    # Ensure the dataframe has been transformed by transform(); otherwise, user should call transform first.
    expected = ['beauty', 'beauty_sq', 'eval', 'female', 'age', 'log_students', 'tenure', 'division', 'credits', 'native', 'minority', 'prof']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"The dataframe is missing required columns for modeling: {missing}. Please run transform().")

    # Formula: linear and quadratic beauty, interaction with female, and covariates (categorical via C()).
    formula = (
        'eval ~ beauty + beauty_sq + female + beauty:female + age + log_students '
        '+ C(tenure) + C(division) + C(credits) + C(native) + C(minority)'
    )

    # Fit OLS
    ols_model = smf.ols(formula, data=df).fit()

    # Compute clustered robust covariance by professor id (prof)
    # If there are fewer clusters than desirable, statsmodels will still compute but inference should be cautious.
    clustered_results = ols_model.get_robustcov_results(cov_type='cluster', groups=df['prof'])

    # Return the results object with clustered covariance
    return clustered_results


