from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to create the predictors used in the statistical model.

    Produces the following new columns (exact names used by the model):
      - log_size_ratio: log(n_focal / n_other)
      - rel_size: n_focal - n_other (auxiliary)
      - rel_males: m_focal - m_other
      - rel_females: f_focal - f_other
      - n_total: n_focal + n_other
      - rel_dist: dist_other - dist_focal
      - Location: categorical label ('FocalHome', 'OtherHome', 'Neutral') based on rel_dist threshold
      - size_by_loc: interaction term log_size_ratio * rel_dist

    The function drops rows with missing values in the key columns required for these computations.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Ensure numeric columns where expected (coerce errors to NaN)
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing critical values
    df = df.dropna(subset=required_cols)

    # Compute relative size measures
    # Use log ratio to capture proportional advantage; add small constant guard if needed
    # but here sizes are positive integers (5-13) so direct log is fine.
    df['log_size_ratio'] = np.log(df['n_focal'] / df['n_other'])
    df['rel_size'] = df['n_focal'] - df['n_other']

    # Sex composition differences
    df['rel_males'] = df['m_focal'] - df['m_other']
    df['rel_females'] = df['f_focal'] - df['f_other']

    # Total group-size context
    df['n_total'] = df['n_focal'] + df['n_other']

    # Location: relative distance to home-center. Positive rel_dist indicates focal is closer to its center
    # than the other is to its center (i.e., location advantage for focal).
    df['rel_dist'] = df['dist_other'] - df['dist_focal']

    # Create a simple 3-level location label. Threshold chosen to reduce classification noise; 50 m is a reasonable
    # heuristic given the range of distances in the dataset. Users can adjust threshold as a sensitivity check.
    threshold_m = 50.0
    df['Location'] = 'Neutral'
    df.loc[df['rel_dist'] > threshold_m, 'Location'] = 'FocalHome'
    df.loc[df['rel_dist'] < -threshold_m, 'Location'] = 'OtherHome'
    df['Location'] = df['Location'].astype('category')

    # Interaction term used in the model (numeric interaction between size advantage and location advantage)
    df['size_by_loc'] = df['log_size_ratio'] * df['rel_dist']

    # Ensure dyad is integer index (used for clustering standard errors)
    df['dyad'] = df['dyad'].astype(int)

    # Keep only columns relevant for downstream modeling and diagnostics
    keep_cols = [
        'win', 'log_size_ratio', 'rel_size', 'rel_males', 'rel_females',
        'n_total', 'rel_dist', 'Location', 'size_by_loc', 'dyad',
        'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other',
        'focal', 'other'
    ]
    # Some columns (focal/other) might be missing in some datasets; keep intersection
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting the probability that the focal group wins ('win' == 1).

    Model specification (primary):
      logit(win) ~ log_size_ratio + rel_dist + size_by_loc + rel_males + rel_females + n_total

    We compute cluster-robust standard errors clustered on 'dyad' to account for non-independence
    within dyads (repeated encounters between the same pair of groups).

    Returns the fitted results object with clustered robust covariance if possible.
    """
    # Columns used in the regression
    predictors = [
        'log_size_ratio',
        'rel_dist',
        'size_by_loc',
        'rel_males',
        'rel_females',
        'n_total'
    ]

    # Ensure predictors are present
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        raise ValueError(f"Missing required predictor columns in transformed dataframe: {missing}")

    # Prepare X and y
    X = df[predictors].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['win'].astype(float)

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    try:
        res = logit_model.fit(disp=False)
    except Exception as e:
        # If fitting fails, raise with a helpful message
        raise RuntimeError(f"Logit model failed to converge: {e}")

    # Try computing cluster-robust standard errors by dyad
    if 'dyad' in df.columns:
        try:
            res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
            return res_clust
        except Exception:
            # If robust clustering fails, return the original results object
            return res
    else:
        return res


