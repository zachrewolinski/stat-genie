from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/replace_and_positive_statement_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for modeling. Produces the following required columns:
      - beauty_z: standardized beauty score (mean 0, sd 1)
      - eval: dependent variable (student evaluation)
      - age: numeric
      - gender_female: binary indicator (1 if female, 0 otherwise)
      - minority_yes: binary indicator (1 if minority == 'yes')
      - credits_single: binary indicator (1 if credits == 'single')
      - division_upper: binary indicator (1 if division == 'upper')
      - native_yes: binary indicator (1 if native == 'yes')
      - tenure_yes: binary indicator (1 if tenure == 'yes')
      - log_students: log(number of students who participated)
      - prof: professor identifier (kept as-is for clustering/grouping)

    Drops rows with missing key variables.
    """
    # Ensure a copy
    df = df.copy()

    # Columns required for the analysis
    required_cols = ['beauty', 'eval', 'age', 'gender', 'minority', 'credits', 'division', 'native', 'tenure', 'students', 'prof']

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Standardize beauty for interpretability (z-score)
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std()

    # Dependent variable: ensure numeric
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')

    # Controls: create binary indicators with explicit column names
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['gender_female'] = (df['gender'].astype(str).str.lower() == 'female').astype(int)
    df['minority_yes'] = (df['minority'].astype(str).str.lower() == 'yes').astype(int)
    df['credits_single'] = (df['credits'].astype(str).str.lower() == 'single').astype(int)
    df['division_upper'] = (df['division'].astype(str).str.lower() == 'upper').astype(int)
    df['native_yes'] = (df['native'].astype(str).str.lower() == 'yes').astype(int)
    df['tenure_yes'] = (df['tenure'].astype(str).str.lower() == 'yes').astype(int)

    # Class size: use log of number of students who participated to reduce skew
    # add small constant if needed (but min(students) in dataset > 0)
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['log_students'] = np.log(df['students'].replace(0, np.nan))

    # After transformations drop any rows that became NA
    keep_cols = ['beauty_z', 'eval', 'age', 'gender_female', 'minority_yes', 'credits_single', 'division_upper', 'native_yes', 'tenure_yes', 'log_students', 'prof']
    df = df.dropna(subset=keep_cols)

    # Optionally reduce dataframe to only the required columns for modeling to avoid accidental usage of other columns
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary OLS model and two robustness checks:
      1) OLS with standard errors clustered by professor (prof)
      2) Linear mixed effects model with random intercept for professor

    Returns a dictionary with fitted model objects and results summaries.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Formula: main effect of beauty, controlling for covariates
    formula = (
        'eval ~ beauty_z + age + gender_female + minority_yes + credits_single '
        '+ division_upper + native_yes + tenure_yes + log_students'
    )

    # Fit OLS
    ols_model = smf.ols(formula, data=df).fit()

    # Clustered (by prof) robust standard errors
    # If prof is numeric but represents groups this still works. Ensure groups variable exists.
    clustered_results = ols_model.get_robustcov_results(cov_type='cluster', groups=df['prof'])

    # Mixed effects model: random intercept by professor
    # Use MixedLM for random intercept; fall back gracefully if convergence issues occur
    mixed_results = None
    try:
        md = sm.MixedLM.from_formula(formula, groups=df['prof'], data=df)
        mixed_results = md.fit(reml=False)
    except Exception as e:
        # If mixed model fails to converge, capture the exception in the returned object
        mixed_results = {'error': str(e)}

    # Package results
    results = {
        'ols_model': ols_model,                    # statsmodels RegressionResultsWrapper
        'ols_clustered': clustered_results,        # clustered robust results
        'mixedlm': mixed_results                   # MixedLMResults or error dict
    }

    return results


