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
    Transformations performed:
    - Drop rows with missing values in variables required for the main analysis.
    - Create a standardized beauty measure 'beauty_z'.
    - Create a logged class-size measure 'students_log' = log1p(students).
    - Ensure categorical columns are of dtype 'category' so that formulas using C(...) work predictably.

    Final dataframe contains at least the following columns used in modeling:
    ['eval', 'beauty_z', 'age', 'students_log', 'gender', 'minority', 'tenure', 'native', 'division', 'credits', 'prof']
    """

    df = df.copy()

    # Required columns
    required_cols = ['eval', 'beauty', 'age', 'students', 'gender', 'minority', 'tenure', 'native', 'division', 'credits', 'prof']
    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Standardize beauty (z-score). The raw 'beauty' is already mean-centered in the original dataset description,
    # but standardizing helps with coefficient interpretation and comparability across models.
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1.0
    df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Log-transform student count to reduce skew (use log1p to be safe)
    df['students_log'] = np.log1p(df['students'])

    # Ensure categorical columns are typed as category (keeps original string labels)
    for cat in ['gender', 'minority', 'tenure', 'native', 'division', 'credits']:
        if cat in df.columns:
            df[cat] = df[cat].astype('category')

    # Ensure prof is an integer identifier (keeps original values but cast to int if possible)
    try:
        df['prof'] = df['prof'].astype(int)
    except Exception:
        # fall back to categorical codes if non-integer
        df['prof'] = df['prof'].astype('category').cat.codes

    # Final safety: keep only needed columns (but preserve any extras if present)
    final_cols = ['eval', 'beauty_z', 'beauty', 'age', 'students', 'students_log', 'gender', 'minority', 'tenure', 'native', 'division', 'credits', 'prof']
    present_final_cols = [c for c in final_cols if c in df.columns]
    return df[present_final_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs two main analyses to assess the relationship between instructor beauty and student evaluation scores:
    1) OLS with relevant controls and cluster-robust standard errors at the professor level.
    2) OLS with professor fixed effects (adds C(prof)) as a robustness check that uses within-professor variation across courses.

    Returns a dictionary with fitted results objects for both models.
    """

    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Make a copy to avoid side-effects
    df = df.copy()

    # Ensure the key columns are present
    required = ['eval', 'beauty_z', 'age', 'students_log', 'gender', 'minority', 'tenure', 'native', 'division', 'credits', 'prof']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Base formula (controls included). C(...) tells patsy/statsmodels to create dummies for categorical vars.
    formula = (
        'eval ~ beauty_z + age + students_log '
        '+ C(gender) + C(minority) + C(tenure) + C(native) + C(division) + C(credits)'
    )

    # Model 1: OLS with cluster-robust SEs by professor (accounts for multiple observations per instructor)
    ols = smf.ols(formula, data=df)
    ols_res = ols.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Model 2: OLS with professor fixed effects (controls for any time-invariant instructor characteristics)
    # Note: including C(prof) uses many dummies; this estimates within-professor effects but requires variation in beauty within prof
    formula_fe = formula + ' + C(prof)'
    fe = smf.ols(formula_fe, data=df)
    fe_res = fe.fit()

    # Optional diagnostics: coefficient of primary interest and its clustered SE
    beauty_coef = ols_res.params.get('beauty_z', float('nan'))
    beauty_se_cluster = ols_res.bse.get('beauty_z', float('nan'))

    results = {
        'ols_cluster': ols_res,
        'fe_ols': fe_res,
        'beauty_coef_ols_cluster': float(beauty_coef),
        'beauty_se_ols_cluster': float(beauty_se_cluster),
        'formula': formula,
        'n_obs': int(ols_res.nobs),
        'n_professors': int(df['prof'].nunique())
    }

    return results


