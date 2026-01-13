from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure key columns are numeric
    num_cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing outcome or essential predictors
    df = df.dropna(subset=['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'])

    # Compute relative size measures
    # rel_size: difference in total group size (focal - other)
    df['rel_size'] = df['n_focal'] - df['n_other']
    # male and female differences
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Compute location advantage: other distance minus focal distance.
    # Positive => focal is closer to its home center than other is to its.
    df['dist_adv'] = df['dist_other'] - df['dist_focal']

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for interpretability.
    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)

    df['rel_size_z'] = zscore(df['rel_size'])
    df['dist_adv_z'] = zscore(df['dist_adv'])
    df['male_diff_z'] = zscore(df['male_diff'])
    df['female_diff_z'] = zscore(df['female_diff'])

    # Ensure dyad is an integer/categorical column used as a control in modeling
    df['dyad'] = df['dyad'].astype('category')

    # Final check: keep only rows with finite standardized predictors
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['rel_size_z', 'dist_adv_z', 'male_diff_z', 'female_diff_z'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Expected columns in df: 'win', 'rel_size_z', 'dist_adv_z', 'male_diff_z', 'female_diff_z', 'dyad'

    # Fit a logistic regression (GLM with binomial family) predicting probability focal wins.
    # Include interaction between relative size and location advantage to test whether the effect
    # of group size depends on contest location. Include dyad fixed effects to control
    # for pair-specific heterogeneity.
    formula = 'win ~ rel_size_z * dist_adv_z + male_diff_z + female_diff_z + C(dyad)'

    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = model.fit()

    # It's useful to also provide clustered robust standard errors by dyad if desired.
    # Attempt to compute cluster-robust covariances; if it fails, fall back to default.
    try:
        clustered = results.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        clustered = results

    # Return the fitted results with cluster-robust SEs when available
    return clustered


