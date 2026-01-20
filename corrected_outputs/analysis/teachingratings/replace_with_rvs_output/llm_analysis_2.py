from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/replace_with_rvs_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh/Parker teaching-evaluations dataset into a dataframe suitable for modeling.
    Steps:
    - Drop rows missing the key variables (beauty or eval).
    - Ensure categorical fields have category dtype.
    - Create standardized beauty (beauty_z) for easier interpretation of coefficients.
    - Create log_students as log(number of students who participated) to reduce skew.
    - Ensure professor id is integer and other columns are in expected formats.
    Returns the transformed dataframe containing at least the columns referenced in the model.
    """
    # Drop rows with missing main variables
    df = df.copy()
    df = df.dropna(subset=['beauty', 'eval'])

    # Convert obvious categorical fields to category dtype (keeps original string labels)
    for cat_col in ['gender', 'minority', 'division', 'native', 'tenure', 'credits']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')

    # Standardize beauty (z-score)
    # If beauty has zero variance for some reason, guard against division by zero
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1.0
    df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Create log of students (participants). students is >=5 in this dataset, but guard anyway
    # Replace non-positive or missing students with NaN before log
    df['log_students'] = np.log(df['students'].replace({0: np.nan}))

    # Ensure prof is integer (used for clustering)
    if 'prof' in df.columns:
        try:
            df['prof'] = pd.to_numeric(df['prof'], errors='coerce').astype('Int64')
        except Exception:
            # fallback: keep as-is
            pass

    # Final: keep columns required for modeling plus original eval and beauty for reference
    required_cols = [
        'eval', 'beauty', 'beauty_z', 'age', 'gender', 'minority', 'tenure', 'log_students',
        'division', 'native', 'credits', 'prof'
    ]
    # Some columns may not exist in all inputs; keep intersection
    cols_to_return = [c for c in required_cols if c in df.columns]
    return df[cols_to_return]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Estimate the effect of instructor beauty on teaching evaluations.

    Returns a dictionary containing:
    - 'baseline': OLS of eval on beauty_z (no controls),
    - 'full_cluster': OLS of eval on beauty_z + controls with standard errors clustered by professor (prof).

    Both models are returned as fitted statsmodels results objects.
    """
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Baseline model: eval on beauty only
    baseline_vars = ['eval', 'beauty_z']
    df_baseline = df.dropna(subset=baseline_vars)
    model_baseline = smf.ols('eval ~ beauty_z', data=df_baseline)
    res_baseline = model_baseline.fit()

    # Full model with controls
    # Define formula including categorical controls using C(...) to ensure proper dummying
    formula_parts = [
        'beauty_z',
        'age',
        'log_students',
        'C(gender)',
        'C(minority)',
        'C(tenure)',
        'C(division)',
        'C(native)',
        'C(credits)'
    ]
    formula = 'eval ~ ' + ' + '.join(formula_parts)

    # Drop rows missing any variables used in the full model or missing prof for clustering
    required_full = ['eval', 'beauty_z', 'age', 'log_students', 'gender', 'minority', 'tenure', 'division', 'native', 'credits', 'prof']
    # Keep only existing columns for dropna (in case some controls are absent)
    existing_required = [c for c in required_full if c in df.columns]
    df_full = df.dropna(subset=existing_required)

    model_full = smf.ols(formula, data=df_full)

    # If prof exists, cluster standard errors by prof; otherwise use robust HC1 se
    if 'prof' in df_full.columns and not df_full['prof'].isnull().all():
        try:
            res_full = model_full.fit(cov_type='cluster', cov_kwds={'groups': df_full['prof']})
        except Exception:
            # fallback to robust standard errors if clustering fails
            res_full = model_full.fit(cov_type='HC1')
    else:
        res_full = model_full.fit(cov_type='HC1')

    results = {
        'baseline': res_baseline,
        'full_cluster': res_full
    }
    return results


