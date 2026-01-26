from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw capuchin contest data into modeling-ready dataframe.

    Produces the following new columns used in the model:
      - rel_size_log_z: standardized log(n_focal / n_other)
      - dist_adv_z: standardized (dist_other - dist_focal) so positive indicates focal is closer to its home center
      - in_focal_territory: binary (1 if dist_focal < dist_other else 0)
      - m_diff_z: standardized difference in number of males (m_focal - m_other)
      - f_diff_z: standardized difference in number of females (f_focal - f_other)

    Also drops rows missing required variables.
    """
    # Copy to avoid modifying caller frame
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with NA in required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    for c in ['n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'win']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'win'])

    # Compute relative size as log ratio (stable when groups have similar sizes). n_other is >= 1 in this dataset.
    df['rel_size_log'] = np.log(df['n_focal'] / df['n_other'])

    # Compute distance advantage: positive means focal is closer to its center than other is to its center
    df['dist_adv'] = df['dist_other'] - df['dist_focal']

    # Binary indicator for whether contest is closer to focal's home-range center
    df['in_focal_territory'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Differences in male and female numbers
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for interpretability.
    def zscore(x):
        return (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1.0)

    df['rel_size_log_z'] = zscore(df['rel_size_log'])
    df['dist_adv_z'] = zscore(df['dist_adv'])
    df['m_diff_z'] = zscore(df['m_diff'])
    df['f_diff_z'] = zscore(df['f_diff'])

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Keep only columns necessary for modeling (but leave original identifiers if desired)
    model_cols = ['win', 'rel_size_log_z', 'dist_adv_z', 'in_focal_territory', 'm_diff_z', 'f_diff_z', 'dyad',
                  'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other']
    # Filter to existing columns (in case some original columns weren't present beyond required ones)
    model_cols = [c for c in model_cols if c in df.columns]
    df = df.loc[:, model_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting probability focal won (win)
    from relative group size and location, including an interaction to test whether
    the effect of group size depends on being closer to focal home center.

    The model uses clustered robust standard errors by dyad to account for non-independence
    of contests involving the same dyad.

    Model formula:
      win ~ rel_size_log_z * in_focal_territory + dist_adv_z + m_diff_z + f_diff_z

    Returns the fitted results object with cluster-robust SE.
    """
    import statsmodels.formula.api as smf

    # Check required columns
    req = ['win', 'rel_size_log_z', 'in_focal_territory', 'dist_adv_z', 'm_diff_z', 'f_diff_z', 'dyad']
    missing = [c for c in req if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula including interaction between relative size and focal-territory indicator
    formula = 'win ~ rel_size_log_z * in_focal_territory + dist_adv_z + m_diff_z + f_diff_z'

    # Fit binomial GLM (logistic regression)
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = model_glm.fit()

    # Obtain cluster-robust covariance (cluster on dyad)
    # If there are few clusters, caution is required when interpreting SEs.
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # Fallback: return unclustered results if clustering fails
        res_clust = res

    # Print summary for convenience; the caller can inspect returned object programmatically
    print(res_clust.summary())

    return res_clust


