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
    Transform the raw Hamermesh & Parker classroom dataset into a modeling-ready dataframe.

    Produces the following columns (used in the model):
      - eval: dependent variable (kept as-is)
      - beauty_z: standardized beauty score (IV)
      - age_c: age centered at the sample mean
      - gender_female: dummy (1 if female, 0 if male)
      - minority_yes: dummy (1 if minority == 'yes', 0 if 'no')
      - native_yes: dummy (1 if native == 'yes')
      - tenure_yes: dummy (1 if tenure == 'yes')
      - division_upper: dummy (1 if division == 'upper')
      - credits_single: dummy (1 if credits == 'single')
      - log_students: log of 'students' (numeric) to reduce skew
      - prof: professor id (as int)

    Rows with missing values in key columns (eval, beauty, students, prof) are dropped.
    """
    df = df.copy()

    # Basic required columns
    required = ['eval', 'beauty', 'students', 'prof']
    # Drop rows missing any required column
    df = df.dropna(subset=required)

    # Create standardized beauty
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std(ddof=0)

    # Center age (leave NaN if missing); keep original missing handling conservative
    if 'age' in df.columns:
        df['age_c'] = df['age'] - df['age'].mean()
    else:
        df['age_c'] = np.nan

    # Binary dummies from categorical factors
    # gender: 'male' or 'female'
    df['gender_female'] = (df['gender'] == 'female').astype(int)

    # minority: 'yes' / 'no'
    df['minority_yes'] = (df['minority'] == 'yes').astype(int)

    # native: 'yes' / 'no'
    df['native_yes'] = (df['native'] == 'yes').astype(int)

    # tenure: 'yes' / 'no'
    df['tenure_yes'] = (df['tenure'] == 'yes').astype(int)

    # division: 'lower' / 'upper'
    df['division_upper'] = (df['division'] == 'upper').astype(int)

    # credits: 'single' / 'more'
    df['credits_single'] = (df['credits'] == 'single').astype(int)

    # Log transform of number of students participating in evaluation to reduce skew
    # students guaranteed >=5 in schema; guard against zeros
    df['log_students'] = np.log(df['students'].replace(0, np.nan))

    # Ensure professor id is integer (grouping variable)
    df['prof'] = df['prof'].astype(int)

    # Keep only the columns needed for modeling plus the original eval and beauty for reference
    keep_cols = [
        'eval', 'beauty', 'beauty_z', 'age_c', 'gender_female', 'minority_yes',
        'native_yes', 'tenure_yes', 'division_upper', 'credits_single', 'log_students', 'prof'
    ]
    # Some columns may not exist if the dataset is missing them; intersect
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two models testing the effect of instructor beauty on student evaluations:
      1) OLS with a battery of controls and cluster-robust standard errors by professor.
      2) Linear mixed-effects model with professor random intercept to account for multiple courses per professor.

    Returns a dictionary with fitted result objects (OLS and MixedLM) and prints brief summaries.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns exist
    required_cols = ['eval', 'beauty_z', 'age_c', 'gender_female', 'minority_yes',
                     'native_yes', 'tenure_yes', 'division_upper', 'credits_single',
                     'log_students', 'prof']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop any rows with missing values in the modeling variables
    model_df = df.dropna(subset=required_cols).copy()

    # Define formula
    formula = (
        'eval ~ beauty_z + age_c + gender_female + minority_yes + native_yes + '
        'tenure_yes + division_upper + credits_single + log_students'
    )

    # 1) OLS with cluster-robust SE by professor (accounts for within-prof correlation)
    ols_model = smf.ols(formula, data=model_df).fit()
    # Clustered standard errors by professor id
    try:
        ols_clustered = ols_model.get_robustcov_results(cov_type='cluster',
                                                       cov_kwds={'groups': model_df['prof']})
    except Exception:
        # fallback to HC3 if clustering fails for any reason
        ols_clustered = ols_model.get_robustcov_results(cov_type='HC3')

    # 2) Mixed effects model with random intercept for professor
    # This accounts for multiple courses taught by the same professor
    mixed_result = None
    try:
        mixed = sm.MixedLM.from_formula(formula, groups='prof', data=model_df)
        mixed_result = mixed.fit(reml=False)
    except Exception as e:
        # If MixedLM fails to converge or errors, keep None and include the exception message
        mixed_result = e

    # Print concise summaries
    print('OLS (cluster-robust SE by prof) coefficient for beauty_z:')
    print(ols_clustered.summary().tables[1])

    if isinstance(mixed_result, Exception):
        print('\nMixedLM failed:', mixed_result)
    else:
        print('\nMixedLM random intercept model summary (coefficients):')
        print(mixed_result.summary())

    # Return results so caller can inspect programmatically
    return {
        'ols_model': ols_model,
        'ols_clustered': ols_clustered,
        'mixedlm_result': mixed_result,
        'model_df': model_df
    }


