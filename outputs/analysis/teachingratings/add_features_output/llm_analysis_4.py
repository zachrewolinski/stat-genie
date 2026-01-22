from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Hamermesh & Parker classroom dataset into analytical dataframe.

    Produces:
      - centered beauty (beauty_c) and its square (beauty_sq)
      - binary controls (gender_female, minority_yes, tenure_yes, native_yes,
        credits_single, division_upper)
      - log_students (log of 'students')
      - ensures eval and prof are present

    Returns dataframe with all columns used in modeling.
    """
    df = df.copy()

    # Keep only rows with key variables present
    # eval (DV), beauty (IV), prof (grouping), students (for log), age used as control
    key_cols = ['eval', 'beauty', 'prof', 'students', 'age']
    df = df.dropna(subset=key_cols)

    # Normalize/clean string categorical columns (guard against leading/trailing spaces)
    # Many factor columns appear as strings like 'yes'/'no' or 'male'/'female'
    for col in ['gender', 'minority', 'tenure', 'native', 'credits', 'division']:
        if col in df.columns:
            # convert entries to lowercase strings and strip whitespace, keep nans
            df[col] = df[col].astype(str).str.strip().str.lower().replace({'nan': np.nan})

    # Binary indicators
    if 'gender' in df.columns:
        df['gender_female'] = (df['gender'] == 'female').astype(int)
    else:
        df['gender_female'] = 0

    if 'minority' in df.columns:
        df['minority_yes'] = (df['minority'] == 'yes').astype(int)
    else:
        df['minority_yes'] = 0

    if 'tenure' in df.columns:
        df['tenure_yes'] = (df['tenure'] == 'yes').astype(int)
    else:
        df['tenure_yes'] = 0

    if 'native' in df.columns:
        df['native_yes'] = (df['native'] == 'yes').astype(int)
    else:
        df['native_yes'] = 0

    if 'credits' in df.columns:
        df['credits_single'] = (df['credits'] == 'single').astype(int)
    else:
        df['credits_single'] = 0

    if 'division' in df.columns:
        df['division_upper'] = (df['division'] == 'upper').astype(int)
    else:
        df['division_upper'] = 0

    # Log-transform class size (use participants in evaluation 'students')
    # Guard against non-positive counts
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df.loc[df['students'] <= 0, 'students'] = np.nan
    df['log_students'] = np.log(df['students'])

    # Center the beauty variable for interpretation and build a squared term to check nonlinearity
    df['beauty'] = pd.to_numeric(df['beauty'], errors='coerce')
    # If beauty is constant or missing for many, centering will still work but modeler should inspect
    beauty_mean = df['beauty'].mean()
    df['beauty_c'] = df['beauty'] - beauty_mean
    df['beauty_sq'] = df['beauty_c'] ** 2

    # Ensure eval is numeric
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')

    # Keep only rows with no missing values on final modeling columns
    model_cols = [
        'eval', 'beauty_c', 'beauty_sq', 'age', 'gender_female', 'minority_yes',
        'tenure_yes', 'native_yes', 'credits_single', 'division_upper', 'log_students', 'prof'
    ]
    df = df.dropna(subset=model_cols)

    # Ensure prof is integer (group identifier)
    # If prof is not numeric, keep as is (MixedLM accepts grouping labels), but cast to int when possible
    try:
        df['prof'] = pd.to_numeric(df['prof'], errors='ignore')
    except Exception:
        pass

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits multiple specifications to estimate the effect of instructor beauty on student evaluations.

    Models returned:
      - OLS (plain) with controls
      - OLS with cluster-robust standard errors clustered by 'prof'
      - Linear mixed effects model with random intercept by 'prof' (hierarchical model)

    Returns a dict with the fitted result objects for further inspection.
    """
    df = df.copy()

    # Define predictors used in the linear predictor
    predictors = [
        'beauty_c', 'beauty_sq', 'age', 'gender_female', 'minority_yes',
        'tenure_yes', 'native_yes', 'credits_single', 'division_upper', 'log_students'
    ]

    # Drop any rows with missing values in predictors, DV, or prof
    df = df.dropna(subset=predictors + ['eval', 'prof'])

    # Construct design matrix
    X = sm.add_constant(df[predictors], has_constant='add')
    y = df['eval']

    results = {}

    # 1) OLS
    ols_model = sm.OLS(y, X)
    ols_res = ols_model.fit()
    results['ols'] = ols_res

    # 2) OLS with cluster-robust SEs (clusters = prof)
    # Use get_robustcov_results for clustering
    try:
        ols_cluster_res = ols_res.get_robustcov_results(cov_type='cluster', groups=df['prof'])
        results['ols_cluster'] = ols_cluster_res
    except Exception as e:
        # If clustering fails, store the exception in the results for debugging
        results['ols_cluster_error'] = str(e)

    # 3) Mixed effects model (random intercept for prof)
    # MixedLM requires exog and groups; include same predictors and a constant
    try:
        # For MixedLM, provide the same exogenous regressors (including constant)
        md = sm.MixedLM(endog=y, exog=X, groups=df['prof'])
        mixed_res = md.fit(reml=False, method='lbfgs')
        results['mixedlm'] = mixed_res
    except Exception as e:
        results['mixedlm_error'] = str(e)

    # Provide brief printed summaries for convenience (caller may print or inspect objects)
    # Note: The result objects themselves are returned for full inspection
    try:
        print('\n=== OLS summary ===')
        print(ols_res.summary())
    except Exception:
        pass

    try:
        if 'ols_cluster' in results:
            print('\n=== OLS (clustered SE by prof) summary ===')
            print(results['ols_cluster'].summary())
    except Exception:
        pass

    try:
        if 'mixedlm' in results:
            print('\n=== MixedLM (random intercept by prof) summary ===')
            print(results['mixedlm'].summary())
    except Exception:
        pass

    return results


