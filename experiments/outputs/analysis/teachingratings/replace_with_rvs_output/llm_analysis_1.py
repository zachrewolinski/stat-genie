from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/replace_with_rvs_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh & Parker classroom dataset into analysis-ready variables.

    Outputs (added/modified columns):
      - beauty_c: mean-centered beauty score (float)
      - female: binary (1=female, 0=male)
      - minority_yes: binary (1=yes, 0=no)
      - credits_more: binary (1='more', 0='single')
      - division_upper: binary (1='upper', 0='lower')
      - native_yes: binary (1=yes, 0=no)
      - tenure_yes: binary (1=yes, 0=no)
      - ln_students: natural log of 'students'
      - prof: integer professor id (kept for clustering / random effects)

    Notes: Drops rows missing the primary outcome (eval) or primary predictor (beauty).
    """
    df = df.copy()

    # Drop rows missing key variables
    df = df.dropna(subset=['eval', 'beauty'])

    # Center beauty for interpretability (keeps original 'beauty' column too)
    df['beauty_c'] = df['beauty'].astype(float) - float(df['beauty'].astype(float).mean())

    # Binary encodings for categorical controls; robust to capitalization / stray whitespace
    df['female'] = (df['gender'].astype(str).str.strip().str.lower() == 'female').astype(int)
    df['minority_yes'] = (df['minority'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    df['credits_more'] = (df['credits'].astype(str).str.strip().str.lower() == 'more').astype(int)
    df['division_upper'] = (df['division'].astype(str).str.strip().str.lower() == 'upper').astype(int)
    df['native_yes'] = (df['native'].astype(str).str.strip().str.lower() == 'yes').astype(int)
    df['tenure_yes'] = (df['tenure'].astype(str).str.strip().str.lower() == 'yes').astype(int)

    # Log-transform of student count to reduce skew / large-class influence
    # Ensure numeric type first; students has min > 0 in schema
    df['ln_students'] = np.log(df['students'].astype(float))

    # Ensure professor id is integer (used for clustering and random effects)
    # If 'prof' has missing or non-integer values this will raise; keep original otherwise
    df['prof'] = df['prof'].astype(int)

    # Return only the columns necessary for modeling plus original eval/beauty if desired
    # but keep the entire dataframe to preserve other metadata
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit models to estimate the effect of instructor beauty on student evaluation scores.

    Models returned:
      - OLS with controls and an interaction between beauty and gender; cluster-robust SEs by professor id.
      - Linear mixed-effects model (random intercept by professor) using the same controls (no interaction in mixed model by default).

    Returns a dictionary with keys 'ols' and 'mixedlm' containing the fitted results objects.
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Ensure required columns exist
    required_cols = ['eval', 'beauty_c', 'female', 'age', 'minority_yes', 'credits_more',
                     'division_upper', 'native_yes', 'tenure_yes', 'ln_students', 'prof']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # OLS with interaction: beauty effect and whether it differs for female instructors
    formula = (
        'eval ~ beauty_c * female + age + minority_yes + credits_more + '
        'division_upper + native_yes + tenure_yes + ln_students'
    )

    # Fit OLS and obtain cluster-robust standard errors clustered on 'prof'
    ols_res = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Fit a mixed-effects model with a random intercept by professor as a complementary approach
    # This accounts for multiple courses taught by the same professor
    try:
        md = sm.MixedLM.from_formula(
            'eval ~ beauty_c + female + age + minority_yes + credits_more + division_upper + native_yes + tenure_yes + ln_students',
            groups=df['prof'],
            data=df
        )
        mdf = md.fit(reml=False)
    except Exception as e:
        # If the mixed model fails to converge or errors, return None for mixedlm and keep OLS
        mdf = None

    # Return results objects. Users can call .summary() on ols_res and mdf (if not None).
    return {'ols': ols_res, 'mixedlm': mdf}


