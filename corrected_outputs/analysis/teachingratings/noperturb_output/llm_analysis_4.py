from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/noperturb_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to a dataframe ready for modeling.

    Produces the following columns required by the model:
      - eval: dependent variable (kept as-is)
      - beauty_z: standardized beauty rating (z-score)
      - age, students, allstudents: numeric controls
      - gender_F, minority_Y, tenure_Y, native_Y, division_upper, credits_more: binary dummies
      - beauty_gender_interaction: interaction between beauty_z and gender_F
      - prof: instructor id (kept for fixed effects / clustering)

    The function drops rows with missing values in any of the variables used in modeling.
    """
    df = df.copy()

    # Required raw columns
    required_cols = [
        'eval', 'beauty', 'age', 'students', 'allstudents', 'gender',
        'minority', 'tenure', 'native', 'division', 'credits', 'prof'
    ]

    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Standardize beauty to a z-score for interpretability
    # use population std (ddof=0) for stable scaling; fallback to 1 if std == 0
    beauty_std = df['beauty'].std(ddof=0)
    if pd.isna(beauty_std) or beauty_std == 0:
        df['beauty_z'] = df['beauty'] - df['beauty'].mean()
    else:
        df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / beauty_std

    # Binary indicator creation (consistent lowercase comparisons)
    df['gender_F'] = (df['gender'].astype(str).str.lower() == 'female').astype(int)
    df['minority_Y'] = (df['minority'].astype(str).str.lower() == 'yes').astype(int)
    df['tenure_Y'] = (df['tenure'].astype(str).str.lower() == 'yes').astype(int)
    df['native_Y'] = (df['native'].astype(str).str.lower() == 'yes').astype(int)
    df['division_upper'] = (df['division'].astype(str).str.lower() == 'upper').astype(int)
    df['credits_more'] = (df['credits'].astype(str).str.lower() == 'more').astype(int)

    # Ensure numeric columns are numeric and drop rows that failed coercion
    numeric_cols = ['age', 'students', 'allstudents', 'prof', 'eval']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols)

    # Interaction term between standardized beauty and gender
    df['beauty_gender_interaction'] = df['beauty_z'] * df['gender_F']

    # Final sanity: keep only columns that will be used in model
    keep_cols = [
        'eval', 'beauty_z', 'age', 'students', 'allstudents', 'gender_F',
        'minority_Y', 'tenure_Y', 'native_Y', 'division_upper', 'credits_more',
        'beauty_gender_interaction', 'prof'
    ]
    df = df[keep_cols].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating the effect of instructor beauty on teaching evaluations.

    Model specification:
      eval ~ beauty_z + age + students + allstudents + gender_F + minority_Y +
              tenure_Y + native_Y + division_upper + credits_more +
              beauty_gender_interaction + C(prof)

    - C(prof) includes instructor fixed effects (as categorical dummies).
    - Cluster-robust standard errors by instructor 'prof' are used to account for within-instructor correlation.

    Returns the fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    # Ensure the categorical fixed effect is treated properly in the formula
    formula = (
        'eval ~ beauty_z + age + students + allstudents + gender_F + minority_Y + '
        'tenure_Y + native_Y + division_upper + credits_more + beauty_gender_interaction + C(prof)'
    )

    # Fit OLS with cluster-robust SE by professor ID
    model = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    return model


