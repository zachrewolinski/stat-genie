from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/positive_leading_statement_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for analysis. Returns a dataframe with the columns used in the modeling step.

    Transformations performed:
    - Copy dataframe to avoid side effects.
    - Drop rows missing the dependent variable ('eval') or the key independent variable ('beauty').
    - Ensure numeric columns are numeric, categorical columns are category dtype.
    - Compute log_students = log(number of students) to reduce skew.
    - Standardize (z-score) beauty to create 'beauty_z' for interpretable coefficients.

    Final dataframe columns used in modeling: ['beauty','beauty_z','eval','age','gender','minority','division','credits','native','tenure','students','log_students','prof','allstudents']
    """
    df = df.copy()

    # Ensure expected columns exist
    expected_cols = ['beauty', 'eval', 'age', 'gender', 'minority', 'division', 'credits', 'native', 'tenure', 'students', 'prof', 'allstudents']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input dataframe: {missing}")

    # Drop rows missing the dependent variable or main IV
    df = df.dropna(subset=['eval', 'beauty'])

    # Coerce numeric columns
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['allstudents'] = pd.to_numeric(df.get('allstudents', pd.Series(np.nan, index=df.index)), errors='coerce')

    # Convert categorical columns to category dtype (keeps original labels for formula usage)
    for col in ['gender', 'minority', 'division', 'credits', 'native', 'tenure']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Remove any rows that became NA after coercion for crucial columns
    df = df.dropna(subset=['age', 'students'])

    # Create log of class size to reduce skew (add small constant if zero but here students>=5 by schema)
    df['log_students'] = np.log(df['students'])

    # Standardize beauty (z-score) for easier interpretation of coefficients
    if df['beauty'].std(ddof=0) == 0 or np.isnan(df['beauty'].std(ddof=0)):
        df['beauty_z'] = df['beauty'] - df['beauty'].mean()
    else:
        df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std(ddof=0)

    # Keep only relevant columns (preserve additional info like allstudents and prof)
    keep_cols = ['beauty', 'beauty_z', 'eval', 'age', 'gender', 'minority', 'division', 'credits', 'native', 'tenure', 'students', 'log_students', 'prof', 'allstudents']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run a sequence of regression models to estimate the effect of instructor beauty on student evaluations.

    Specifications:
    - Model 1 (m1): bivariate OLS: eval ~ beauty_z
    - Model 2 (m2): OLS controlling for instructor and course covariates, with cluster-robust standard errors clustered at the professor level: eval ~ beauty_z + controls
    - Model 3 (m3): OLS with professor fixed effects (C(prof)) plus controls, with clustering by professor to account for within-prof dependence.

    Returns a dict with fitted statsmodels results objects for each model.
    """
    import statsmodels.formula.api as smf

    # Ensure the DataFrame contains required columns
    required = ['eval', 'beauty_z', 'age', 'gender', 'minority', 'division', 'credits', 'native', 'tenure', 'log_students', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    results = {}

    # Model 1: simple bivariate relationship
    formula_m1 = 'eval ~ beauty_z'
    m1 = smf.ols(formula_m1, data=df).fit()
    results['m1'] = m1

    # Model 2: controls, cluster standard errors by professor
    formula_m2 = 'eval ~ beauty_z + age + C(gender) + C(minority) + C(division) + C(credits) + C(native) + C(tenure) + log_students'
    try:
        m2 = smf.ols(formula_m2, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
    except Exception:
        # Fallback to default OLS if clustering fails for any reason
        m2 = smf.ols(formula_m2, data=df).fit()
    results['m2'] = m2

    # Model 3: professor fixed effects (controls + C(prof)). Because C(prof) can be high-dimensional, clustering by prof remains important.
    formula_m3 = 'eval ~ beauty_z + age + log_students + C(gender) + C(minority) + C(division) + C(credits) + C(native) + C(tenure) + C(prof)'
    try:
        m3 = smf.ols(formula_m3, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
    except Exception:
        # In case of perfect multicollinearity or other issues with fixed effects, drop C(prof) and fall back to m2-like spec
        m3 = smf.ols(formula_m2 + ' + 0', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
    results['m3'] = m3

    # Print short summaries for quick inspection (can be commented out in production)
    try:
        print('\nModel 1 (bivariate)')
        print(m1.summary())
        print('\nModel 2 (controls, clustered SE by prof)')
        print(m2.summary())
        print('\nModel 3 (professor fixed effects, clustered by prof)')
        print(m3.summary())
    except Exception:
        pass

    return results


