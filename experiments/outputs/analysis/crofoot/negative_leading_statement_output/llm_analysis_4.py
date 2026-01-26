from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/negative_leading_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe into the analysis-ready dataframe.

    Produces standardized predictors:
      - size_diff_z: z-scored (n_focal - n_other)
      - home_advantage_z: z-scored (dist_other - dist_focal) [positive => closer to focal home center]
      - m_diff_z: z-scored (m_focal - m_other)
      - total_size_z: z-scored (n_focal + n_other)

    Keeps the binary outcome 'win' and 'dyad' for clustering.
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'win', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Compute relative size (difference) and location advantage
    df['size_diff'] = df['n_focal'] - df['n_other']
    # Also create ratio (not used in main model but kept for diagnostics)
    # Add small epsilon to avoid division by zero (there is no zero-sized group in this dataset, but be safe)
    eps = 1e-6
    df['size_ratio'] = (df['n_focal'] + eps) / (df['n_other'] + eps)
    df['size_ratio_log'] = np.log(df['size_ratio'])

    # Home advantage: positive means contest is closer to focal group's home-range center
    df['home_advantage'] = df['dist_other'] - df['dist_focal']

    # Male difference and total size
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['total_size'] = df['n_focal'] + df['n_other']

    # Standardize continuous predictors (z-score). Use ddof=0 for population SD.
    for col in ['size_diff', 'home_advantage', 'm_diff', 'total_size']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, create zero column to avoid NaNs
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Keep only columns needed for analysis
    keep_cols = ['win', 'dyad', 'size_diff_z', 'home_advantage_z', 'm_diff_z', 'total_size_z',
                 'size_ratio', 'size_ratio_log', 'size_diff', 'home_advantage', 'm_diff', 'total_size']
    df = df[keep_cols]

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting the probability that the focal group wins.

    Main predictors:
      - size_diff_z (standardized focal - other group size)
      - home_advantage_z (standardized: dist_other - dist_focal)

    Controls:
      - m_diff_z (standardized male difference)
      - total_size_z (standardized combined group size)

    We fit a GLM with Binomial family and then compute cluster-robust standard errors clustered by 'dyad' to account
    for non-independence of multiple contests within the same dyad.

    Returns the fitted results object with cluster-robust covariances applied.
    """
    import statsmodels.api as sm

    # Ensure required columns present
    req = ['win', 'dyad', 'size_diff_z', 'home_advantage_z', 'm_diff_z', 'total_size_z']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Design matrix
    X = df[['size_diff_z', 'home_advantage_z', 'm_diff_z', 'total_size_z']]
    X = sm.add_constant(X)
    y = df['win']

    # Fit GLM (logistic)
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Get cluster-robust covariance results clustered by dyad
    # This adjusts standard errors for repeated observations within dyads.
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If for any reason clustering fails, return the original fit but warn the user
        print("Warning: cluster robust covariance estimation failed; returning non-clustered results.")
        res_cluster = res

    # Print a short summary and return the clustered results object
    print(res_cluster.summary())
    return res_cluster


