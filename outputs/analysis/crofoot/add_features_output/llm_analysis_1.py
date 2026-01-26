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
    Transform the raw capuchin intergroup contest data into a dataframe ready for modeling.
    Outputs (kept in final df):
      - win: dependent variable (0/1)
      - rel_size_ratio_z: standardized focal/other group size ratio (IV)
      - focal_home: binary indicator (1 if contest closer to focal center than other center)
      - male_diff_z: standardized difference in number of males (m_focal - m_other)
      - female_diff_z: standardized difference in number of females (f_focal - f_other)
      - dist_diff_z: standardized (dist_other - dist_focal)
      - dyad: dyad ID (for clustering)
    """
    df = df.copy()

    # Required columns for transformation and modeling
    required = [
        'win', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other',
        'dist_focal', 'dist_other', 'dyad'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required)

    # Compute relative size and raw difference
    df['rel_size_ratio'] = df['n_focal'] / df['n_other']
    df['rel_size_diff'] = df['n_focal'] - df['n_other']

    # Sex-composition differences
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Location: positive dist_diff indicates contest is closer to focal group's center
    df['dist_diff'] = df['dist_other'] - df['dist_focal']

    # Binary indicator for focal home advantage (1 if focal is closer to its center than other is to its center)
    df['focal_home'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize continuous predictors (z-scores). Use population std (ddof=0) to avoid small-sample ddof warnings
    cont_cols = ['rel_size_ratio', 'rel_size_diff', 'male_diff', 'female_diff', 'dist_diff']
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, create a zero column to avoid divide-by-zero
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Keep only columns necessary for modeling
    final_cols = [
        'win',
        'rel_size_ratio_z',
        'focal_home',
        'male_diff_z',
        'female_diff_z',
        'dist_diff_z',
        'dyad'
    ]

    # Ensure final columns exist (in case some intermediate columns are missing due to earlier problems)
    for col in final_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column {col} in transformed dataframe but it is missing.")

    # Return a clean dataframe reset index
    return df[final_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting probability the focal group won the contest.
    Model specification:
      win ~ rel_size_ratio_z + focal_home + rel_size_ratio_z:focal_home +
            male_diff_z + female_diff_z + dist_diff_z

    Standard errors are clustered by dyad to account for repeated dyad observations.

    Returns: statsmodels results object with cluster-robust covariance.
    """
    # Prepare design matrix
    X = df[['rel_size_ratio_z', 'focal_home', 'male_diff_z', 'female_diff_z', 'dist_diff_z']].copy()

    # Interaction between relative size and focal_home (tests whether home advantage moderates size effect)
    X['rel_by_home'] = X['rel_size_ratio_z'] * X['focal_home']

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Dependent variable
    y = df['win'].astype(float)

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    res = logit.fit(disp=0)

    # Cluster-robust covariance by dyad
    # Use get_robustcov_results to obtain clustered standard errors
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['dyad'], use_correction=True)
    except Exception:
        # Fallback: if clustering fails (e.g., too few clusters), return unclustered results but warn
        print("Warning: cluster-robust covariance by dyad failed; returning unclustered results.")
        res_clust = res

    # Optionally print summary for quick inspection (comment out if undesirable)
    print(res_clust.summary())

    return res_clust


