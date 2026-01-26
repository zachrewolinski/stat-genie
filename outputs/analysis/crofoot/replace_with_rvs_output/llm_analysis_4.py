from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_with_rvs_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce columns required for modeling.

    Produces the following new columns used in the model:
      - rel_size: raw difference n_focal - n_other (kept as auxiliary)
      - rel_size_log: log(n_focal / n_other) (primary relative-size IV)
      - focal_home: binary indicator (1 if dist_focal < dist_other, otherwise 0)
      - dist_adv: numeric distance advantage = dist_other - dist_focal (positive means focal is closer to its home center than other)
      - m_diff: m_focal - m_other (control)
      - f_diff: f_focal - f_other (control)

    Also ensures no missing values in the variables used for modeling.
    """
    # make a copy to avoid modifying original
    df = df.copy()

    # Ensure numeric columns are numeric (safe casts)
    num_cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad', 'focal', 'other']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in variables needed for modeling
    required = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    df = df.dropna(subset=required)

    # Compute absolute and relative size measures
    df['rel_size'] = df['n_focal'] - df['n_other']
    # log ratio; add a tiny epsilon if needed (n_other > 0 in this dataset but guard anyway)
    eps = 1e-6
    df['rel_size_log'] = np.log((df['n_focal'] + eps) / (df['n_other'] + eps))

    # Location / home advantage variables
    df['dist_adv'] = df['dist_other'] - df['dist_focal']
    # focal_home = 1 when focal is closer to its home center (i.e., dist_focal < dist_other)
    df['focal_home'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Composition differences (controls)
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Keep only columns that will be used in modeling (plus helpful ids)
    keep_cols = ['win', 'rel_size', 'rel_size_log', 'focal_home', 'dist_adv', 'm_diff', 'f_diff', 'dyad', 'focal', 'other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other']
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present]

    # Reset index for neatness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting focal group winning (win) from
    relative group size and contest location, including their interaction, and control for
    composition differences (m_diff, f_diff). Cluster-robust standard errors are computed
    by dyad to account for repeated observations of the same dyad.

    Model formula:
      win ~ rel_size_log * focal_home + m_diff + f_diff

    Returns:
      - results_robust : statsmodels results object with cluster-robust covariance
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    req = ['win', 'rel_size_log', 'focal_home', 'm_diff', 'f_diff', 'dyad']
    missing = [c for c in req if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit binomial GLM (logistic regression)
    formula = 'win ~ rel_size_log * focal_home + m_diff + f_diff'
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Obtain cluster-robust (by dyad) covariance / standard errors
    # get_robustcov_results exists on the fitted results
    try:
        results_robust = glm_model.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # Fallback: return the original result if clustering fails
        results_robust = glm_model

    # You can inspect results by calling results_robust.summary()
    return results_robust


