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
    """
    Transform the raw capuchin contest dataframe into a modeling-ready dataframe.

    Produces standardized (z-scored) predictors and a few derived columns:
    - size_diff_z: standardized n_focal - n_other
    - dist_diff_z: standardized dist_other - dist_focal (positive -> focal closer -> location advantage)
    - m_diff_z: standardized m_focal - m_other
    - FocalCloser: binary indicator dist_focal < dist_other

    Keeps columns: win, size_diff_z, dist_diff_z, m_diff_z, FocalCloser, dyad (and also log_size_ratio_z and f_diff_z for potential supplementary analyses)
    """
    df = df.copy()

    # Required columns for the analysis
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing critical values
    df = df.dropna(subset=required)

    # Ensure proper dtypes
    df['win'] = df['win'].astype(int)
    df['dyad'] = df['dyad'].astype(int)

    # Derived predictors
    df['size_diff'] = df['n_focal'] - df['n_other']
    # log ratio as alternative relative-size measure (keeps numerical stability)
    df['log_size_ratio'] = np.log((df['n_focal'] + 1e-6) / (df['n_other'] + 1e-6))
    # Positive dist_diff means focal is closer to its home-range center than other (advantage)
    df['dist_diff'] = df['dist_other'] - df['dist_focal']
    df['FocalCloser'] = (df['dist_focal'] < df['dist_other']).astype(int)
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Z-score (standardize) continuous predictors to aid interpretation and numerical stability
    for col in ['size_diff', 'log_size_ratio', 'dist_diff', 'm_diff', 'f_diff']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # avoid division by zero
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Final dataframe columns used for modeling
    keep_cols = [
        'win',
        'size_diff_z',
        'log_size_ratio_z',
        'dist_diff_z',
        'FocalCloser',
        'm_diff_z',
        'f_diff_z',
        'dyad'
    ]

    # If any keep_cols missing (e.g., because std was NaN), create them filled with zeros (safe fallback)
    for c in keep_cols:
        if c not in df.columns:
            df[c] = 0

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (GLM with binomial family) predicting probability that the focal group wins.

    Primary model tests main effects of standardized relative group size (size_diff_z) and standardized
    location advantage (dist_diff_z), their interaction, and controls for male difference and focal proximity.

    Cluster-robust standard errors are computed at the dyad level to account for non-independence of contests
    within the same dyad.

    Returns:
      - the fitted GLM result object (statsmodels GLMResults)
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Check required columns
    required = ['win', 'size_diff_z', 'dist_diff_z', 'm_diff_z', 'FocalCloser', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: main effects + interaction between relative size and location advantage
    formula = 'win ~ size_diff_z * dist_diff_z + m_diff_z + FocalCloser'

    # Fit GLM (logistic regression)
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    # Use cluster-robust SEs clustered on dyad
    results = glm_binom.fit(cov_type='cluster', cov_kwds={'groups': df['dyad']})

    # Print a compact summary and return full results object for downstream inspection
    print(results.summary())
    return results


