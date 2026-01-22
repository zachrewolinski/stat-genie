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
    Transform raw dataset into modeling dataframe. Adds dummies, logs, quadratic beauty term,
    and drops rows with missing key fields.

    Output columns (used in the model):
      - beauty (float)
      - eval (float)
      - age (float)
      - gender_female (0/1)
      - minority_yes (0/1)
      - credits_more (0/1)
      - division_upper (0/1)
      - native_yes (0/1)
      - tenure_yes (0/1)
      - students (int)
      - allstudents (int)
      - log_students (float)
      - log_allstudents (float)
      - beauty_sq (float)
      - prof (int)
    """
    df = df.copy()

    # Keep only rows with non-missing DV and IV
    df = df.dropna(subset=['eval', 'beauty'])

    # Ensure numeric columns are numeric
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')
    df['beauty'] = pd.to_numeric(df['beauty'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['allstudents'] = pd.to_numeric(df['allstudents'], errors='coerce')
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Drop any rows that became NaN after coercion
    df = df.dropna(subset=['eval', 'beauty', 'age', 'students', 'allstudents', 'prof'])

    # Binary/dummy encoding for categorical controls. We create clear column names used in modeling.
    # gender: create gender_female (1 if female, 0 if male)
    df['gender_female'] = df['gender'].map({'female': 1, 'male': 0})
    # If other values exist, coerce to NaN then fill with 0 (conservative)
    df['gender_female'] = df['gender_female'].fillna(0).astype(int)

    # minority
    df['minority_yes'] = df['minority'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # credits: 'more' vs 'single' (1 if more)
    df['credits_more'] = df['credits'].map({'more': 1, 'single': 0}).fillna(0).astype(int)

    # division: upper vs lower
    df['division_upper'] = df['division'].map({'upper': 1, 'lower': 0}).fillna(0).astype(int)

    # native (English speaker)
    df['native_yes'] = df['native'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # tenure
    df['tenure_yes'] = df['tenure'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    # Log-transform student counts to reduce skew. Use natural log on positive counts.
    # students and allstudents are positive in the schema (min 5 and 8 respectively)
    df['log_students'] = np.log(df['students'].clip(lower=1))
    df['log_allstudents'] = np.log(df['allstudents'].clip(lower=1))

    # Quadratic term for beauty to allow for non-linear effects.
    df['beauty_sq'] = df['beauty'] ** 2

    # Keep only the columns we will use in the models (and return full df with these columns present)
    model_cols = [
        'beauty', 'beauty_sq', 'eval', 'age', 'gender_female', 'minority_yes', 'credits_more',
        'division_upper', 'native_yes', 'tenure_yes', 'students', 'allstudents', 'log_students',
        'log_allstudents', 'prof'
    ]

    # If any of these are missing because of dataset oddities, add them with NA so downstream code fails loudly
    for c in model_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Final row-wise drop for any remaining missing modeling values
    df = df.dropna(subset=['beauty', 'eval', 'age', 'students', 'allstudents', 'prof'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run two complementary models to estimate the effect of instructor beauty on student evaluations:
      1) OLS with control variables and standard errors clustered by professor (prof)
      2) Linear mixed effects model (random intercept for professor)

    Returns a dictionary with model result objects and short summaries.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Build design matrix X and response y using the exact column names created in transform()
    features = [
        'beauty', 'beauty_sq', 'age', 'gender_female', 'minority_yes', 'credits_more',
        'division_upper', 'native_yes', 'tenure_yes', 'log_students', 'log_allstudents'
    ]

    # Ensure the columns exist
    missing = [c for c in features + ['eval', 'prof'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Prepare X and y
    X = df[features].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['eval'].astype(float)

    # 1) OLS with clustered SE by prof
    ols_model = sm.OLS(y, X)
    # Fit and request clustered standard errors by professor id
    try:
        ols_res = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
    except Exception:
        # Fallback: fit without cluster if something goes wrong
        ols_res = ols_model.fit()

    # 2) Mixed effects model: random intercept by professor
    # MixedLM expects exog with a constant; use same X
    try:
        mixed = sm.MixedLM(endog=y, exog=X, groups=df['prof'])
        mixed_res = mixed.fit(reml=False, method='lbfgs')
    except Exception as e:
        mixed_res = None

    # Prepare concise numeric summary for the beauty coefficient(s)
    def coef_summary(res, name):
        if res is None:
            return { 'model': name, 'status': 'failed' }
        params = res.params
        b_se = res.bse
        pvals = res.pvalues if hasattr(res, 'pvalues') else None
        return {
            'model': name,
            'coef_beauty': float(params.get('beauty', np.nan)),
            'se_beauty': float(b_se.get('beauty', np.nan)),
            'pval_beauty': float(pvals.get('beauty', np.nan)) if pvals is not None else None,
            'coef_beauty_sq': float(params.get('beauty_sq', np.nan)),
            'se_beauty_sq': float(b_se.get('beauty_sq', np.nan)),
            'pval_beauty_sq': float(pvals.get('beauty_sq', np.nan)) if pvals is not None else None
        }

    results = {
        'ols_result': ols_res,
        'mixedlm_result': mixed_res,
        'summary_beauty_ols': coef_summary(ols_res, 'OLS_clustered'),
        'summary_beauty_mixed': coef_summary(mixed_res, 'MixedLM')
    }

    # Additionally print short human-readable summaries
    try:
        print('OLS clustered results (top lines):')
        print(ols_res.summary().tables[0])
        print('\nBeauty coefficient (OLS clustered):', results['summary_beauty_ols'])
    except Exception:
        pass

    if mixed_res is not None:
        try:
            print('\nMixedLM results (top lines):')
            print(mixed_res.summary().tables[0])
            print('\nBeauty coefficient (MixedLM):', results['summary_beauty_mixed'])
        except Exception:
            pass

    return results


