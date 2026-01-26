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
    # Work on a copy
    df = df.copy()

    # Ensure critical numeric columns exist and coerce types
    numeric_cols = ['eval', 'beauty', 'age', 'students', 'allstudents']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the dependent or primary independent variable
    df = df.dropna(subset=['eval', 'beauty'])

    # Compute response rate (students / allstudents). Guard against zero division.
    df['response_rate'] = np.where(
        (df['allstudents'].notna()) & (df['allstudents'] > 0),
        df['students'] / df['allstudents'],
        np.nan
    )

    # Standardize the beauty measure for easier interpretation
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    if pd.isna(beauty_std) or beauty_std == 0:
        # fallback to avoid division by zero
        df['beauty_z'] = df['beauty'] - beauty_mean
    else:
        df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Convert categorical controls to category dtype to be used with C(...) in formulas
    cat_cols = ['gender', 'minority', 'credits', 'division', 'native', 'tenure']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # Ensure prof is integer for clustering / fixed effects
    if 'prof' in df.columns:
        df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Return dataframe containing all variables used in modeling
    required_cols = [
        'eval', 'beauty', 'beauty_z', 'age', 'students', 'allstudents', 'response_rate',
        'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'prof'
    ]
    # Keep only columns that exist to avoid KeyError downstream; but leave dataframe intact
    # (model function will drop missing values as needed).
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    results = {}

    # Work on a copy of the transformed df and drop rows missing columns required for each model
    df_mod = df.copy()

    # Model 1: simple bivariate relationship (beauty -> eval)
    m1_df = df_mod.dropna(subset=['eval', 'beauty_z'])
    m1 = smf.ols('eval ~ beauty_z', data=m1_df).fit(cov_type='HC3')
    results['model_simple'] = m1

    # Model 2: add observed controls and cluster standard errors by professor (prof) if available
    controls = [
        'age', 'students', 'response_rate',
        'C(gender)', 'C(minority)', 'C(credits)', 'C(division)', 'C(native)', 'C(tenure)'
    ]
    formula_controls = ' + '.join(controls)
    formula2 = f'eval ~ beauty_z + {formula_controls}'

    # drop rows with any of the required predictors missing
    required_m2 = ['eval', 'beauty_z', 'age', 'students', 'response_rate']
    # include categorical columns (they may be present but could be NA)
    cat_required = ['gender', 'minority', 'credits', 'division', 'native', 'tenure']
    for c in cat_required:
        if c in df_mod.columns:
            required_m2.append(c)

    m2_df = df_mod.dropna(subset=[c for c in required_m2 if c in df_mod.columns])

    if 'prof' in m2_df.columns and m2_df['prof'].notna().any():
        # cluster by prof when available
        m2 = smf.ols(formula2, data=m2_df).fit(cov_type='cluster', cov_kwds={'groups': m2_df['prof']})
    else:
        m2 = smf.ols(formula2, data=m2_df).fit(cov_type='HC3')
    results['model_controls_clustered'] = m2

    # Model 3: include professor fixed effects to absorb unobserved instructor-level heterogeneity
    # (this will estimate the beauty effect controlling for any time-invariant prof-level factors)
    if 'prof' in df_mod.columns:
        # include prof as categorical fixed effect; drop rows missing prof or beauty_z or eval
        m3_df = df_mod.dropna(subset=['eval', 'beauty_z', 'prof'])
        # Build formula adding professor fixed effects
        formula3 = f'eval ~ beauty_z + {formula_controls} + C(prof)'
        # Fit with robust (HC3) standard errors; clustering by prof isn't meaningful with C(prof) included
        m3 = smf.ols(formula3, data=m3_df).fit(cov_type='HC3')
        results['model_prof_fe'] = m3
    else:
        results['model_prof_fe'] = None

    # Return the fitted results objects (statsmodels RegressionResults). The caller can call .summary().
    return results

