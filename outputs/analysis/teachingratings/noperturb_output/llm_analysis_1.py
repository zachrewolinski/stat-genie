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
    Transform the raw Hamermesh & Parker classroom dataset into a dataframe ready for modeling.

    Produces the following columns used by the model:
      - beauty_z: standardized beauty score (z-score)
      - eval: course teaching evaluation (dependent variable)
      - age: instructor age
      - gender_female, minority_yes, credits_more, division_upper, native_yes, tenure_yes: binary dummies
      - log_students: log(# students who participated in the evaluation)
      - prof: professor identifier (int)

    Rows with missing values in key variables (beauty, eval, prof) are dropped. Rows with missing values in controls are also dropped for the primary analysis here.
    """
    df = df.copy()

    # Ensure expected columns exist
    required_cols = ['beauty', 'eval', 'age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students', 'prof']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    # Drop rows missing the dependent variable, key IV, or grouping variable
    df = df.dropna(subset=['eval', 'beauty', 'prof'])

    # Also drop rows missing controls (simple approach). If desired, one could impute instead.
    df = df.dropna(subset=['age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students'])

    # Standardize beauty for interpretability (z-score)
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std()

    # Create binary indicator variables from categorical factors; explicit mapping to avoid unknown categories
    df['gender_female'] = df['gender'].map({
        'female': 1,
        'male': 0
    })
    df['minority_yes'] = df['minority'].map({
        'yes': 1,
        'no': 0
    })
    df['credits_more'] = df['credits'].map({
        'more': 1,
        'single': 0
    })
    df['division_upper'] = df['division'].map({
        'upper': 1,
        'lower': 0
    })
    df['native_yes'] = df['native'].map({
        'yes': 1,
        'no': 0
    })
    df['tenure_yes'] = df['tenure'].map({
        'yes': 1,
        'no': 0
    })

    # If mapping produced NaNs (unknown categories), drop those rows
    dummy_cols = ['gender_female', 'minority_yes', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes']
    df = df.dropna(subset=dummy_cols)

    # Class size: use log transform to reduce skew
    # Ensure students > 0
    df = df[df['students'] > 0]
    df['log_students'] = np.log(df['students'])

    # Ensure prof is integer (grouping variable)
    try:
        df['prof'] = df['prof'].astype(int)
    except Exception:
        # If prof cannot be cast to int, create categorical codes
        df['prof'] = pd.Categorical(df['prof']).codes

    # Keep only relevant columns for modeling to avoid accidental usage of others
    keep_cols = [
        'beauty', 'beauty_z', 'eval', 'age',
        'gender_female', 'minority_yes', 'credits_more', 'division_upper',
        'native_yes', 'tenure_yes', 'log_students', 'prof'
    ]
    df = df[keep_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit models estimating the effect of instructor beauty (beauty_z) on teaching evaluations (eval).

    Models returned:
      - ols: standard OLS with covariates
      - ols_cluster: OLS with cluster-robust standard errors clustered on professor (prof)
      - mixedlm: linear mixed-effects model with random intercepts for professor

    The input df is expected to be the transformed dataframe produced by transform(...).
    Returns a dict with the fitted result objects (statsmodels results instances).
    """
    import statsmodels.formula.api as smf
    # statsmodels.api already imported as sm; use it for MixedLM

    # Check that required columns are present
    req = ['eval', 'beauty_z', 'age', 'gender_female', 'minority_yes', 'credits_more', 'division_upper', 'native_yes', 'tenure_yes', 'log_students', 'prof']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe missing required columns for modeling: {missing}")

    # Formula: main IV + controls
    formula = 'eval ~ beauty_z + age + gender_female + minority_yes + credits_more + division_upper + native_yes + tenure_yes + log_students'

    # 1) OLS
    ols_res = smf.ols(formula, data=df).fit()

    # 2) OLS with cluster-robust SEs by professor
    # Use get_robustcov_results to obtain clustered standard errors
    try:
        ols_cluster_res = ols_res.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        # fallback: return the original ols if clustering fails
        ols_cluster_res = ols_res

    # 3) Linear mixed effects model with random intercept per professor
    # This accounts for unobserved instructor-level heterogeneity.
    try:
        md = sm.MixedLM.from_formula(formula, groups='prof', data=df)
        mixed_res = md.fit(reml=False)
    except Exception as e:
        mixed_res = None

    results = {
        'ols': ols_res,
        'ols_cluster': ols_cluster_res,
        'mixedlm': mixed_res
    }

    return results


