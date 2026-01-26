from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw DataFrame into the final DataFrame used for modeling.

    Produced columns (used in model):
      - dyad_win: binary outcome (1 focal won, 0 other won)
      - total_focal: total adults in focal group (n_focal + dist_focal)
      - total_other: total adults in other group (other + focal)
      - log_size_ratio: log(total_focal / total_other)
      - log_size_ratio_c: standardized (z-scored) log_size_ratio
      - focal_dist_m: distance (m) of focal group from center of its home range (column 'win' per metadata)
      - focal_dist_m_c: standardized focal_dist_m
      - other_dist_m: distance (m) of other group from center of its home range (column 'm_focal' per metadata)
      - size_x_focaldist: interaction (log_size_ratio_c * focal_dist_m_c)
      - pair_id: dyad pair id (m_other)
      - focal_id: focal group id (n_other)
      - other_id: other group id (dist_other)

    Notes on mappings: the provided metadata labels are inconsistent with column names; the code below follows the dataset column names and uses the descriptive metadata to interpret which columns represent counts and distances.
    """

    df = df.copy()

    # Required raw columns - ensure they exist
    required_cols = ['dyad', 'n_focal', 'other', 'dist_focal', 'focal', 'win', 'm_focal', 'm_other', 'n_other', 'dist_other']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"The following required columns are missing from the input dataframe: {missing}")

    # Outcome
    df['dyad_win'] = df['dyad'].astype(int)

    # Compute total group sizes using male + female counts (per metadata descriptions):
    # total_focal: males in focal (n_focal) + females in focal (dist_focal per metadata)
    # total_other: males in other (other) + females in other (focal per metadata)
    df['total_focal'] = df['n_focal'] + df['dist_focal']
    df['total_other'] = df['other'] + df['focal']

    # Remove impossible rows (zero or missing totals) to avoid division-by-zero
    df = df.dropna(subset=['total_focal', 'total_other', 'win', 'm_focal', 'dyad'])
    df = df[(df['total_other'] > 0) & (df['total_focal'] > 0)]

    # Relative group size: log ratio (focal / other). Add small epsilon for numerical safety.
    eps = 1e-6
    df['log_size_ratio'] = np.log((df['total_focal'] + eps) / (df['total_other'] + eps))

    # Contest location distances (per metadata):
    # 'win' described as distance (m) of focal group from center of its home range
    # 'm_focal' described as distance (m) of other group from the center of its home range
    df['focal_dist_m'] = df['win'].astype(float)
    df['other_dist_m'] = df['m_focal'].astype(float)

    # IDs for clustering and bookkeeping
    df['pair_id'] = df['m_other'].astype(int)
    df['focal_id'] = df['n_other'].astype(int)
    df['other_id'] = df['dist_other'].astype(int)

    # Standardize (z-score) the main continuous predictors for interpretability and to help model convergence
    # Use ddof=0 to match population std as typical in ML; statsmodels is fine with either.
    df['log_size_ratio_c'] = (df['log_size_ratio'] - df['log_size_ratio'].mean()) / (df['log_size_ratio'].std(ddof=0) + eps)
    df['focal_dist_m_c'] = (df['focal_dist_m'] - df['focal_dist_m'].mean()) / (df['focal_dist_m'].std(ddof=0) + eps)

    # Interaction term: relative size x focal distance (standardized interaction)
    df['size_x_focaldist'] = df['log_size_ratio_c'] * df['focal_dist_m_c']

    # Keep only columns necessary for modeling and diagnostics (but preserve in df to return)
    # Ensure integer columns are int dtype
    df['n_focal'] = df['n_focal'].astype(int)
    df['other'] = df['other'].astype(int)

    # Final returned dataframe includes both raw and derived columns for transparency
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic (binomial) regression predicting focal-group win (dyad_win) from
    relative group size, contest location, their interaction, and controls.

    The function uses cluster-robust standard errors clustered on the dyad pair id (pair_id)
    to account for non-independence of observations within the same group pair.

    Returns the fitted GLMResults object.
    """

    # Columns expected in transformed dataframe
    expected = ['dyad_win', 'log_size_ratio_c', 'focal_dist_m_c', 'size_x_focaldist', 'n_focal', 'other', 'pair_id']
    missing = [c for c in expected if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"The following required columns are missing from the transformed dataframe: {missing}")

    # Design matrix
    X_cols = ['log_size_ratio_c', 'focal_dist_m_c', 'size_x_focaldist', 'n_focal', 'other']
    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['dyad_win'].astype(float)

    # Fit GLM with binomial family (logistic regression) and cluster-robust SEs by pair_id
    model = sm.GLM(y, X, family=sm.families.Binomial())
    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['pair_id']})
    except Exception:
        # Fallback to default fit if cluster covariance fails for small sample reasons
        results = model.fit()

    # Print a concise summary (caller can inspect results further)
    print(results.summary())

    return results


