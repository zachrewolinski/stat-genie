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
    Transform the raw capuchin intergroup contest dataframe to create the variables
    needed for the statistical model.

    Creates the following new columns used in modeling:
      - RelSizeDiff: n_focal - n_other (integer)
      - RelSizeRatio: n_focal / n_other (float)
      - m_diff: m_focal - m_other (integer)
      - f_diff: f_focal - f_other (integer)
      - FocalHomeAdv: 1 if dist_focal < dist_other else 0 (int)

    Drops rows with missing values in any of the variables used for modeling.
    """
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric (coerce if needed)
    numeric_cols = ['dist_focal', 'dist_other', 'n_focal', 'n_other',
                    'm_focal', 'm_other', 'f_focal', 'f_other', 'win', 'dyad']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Derive relative size measures
    df['RelSizeDiff'] = df['n_focal'] - df['n_other']
    # Use ratio as an alternative scaling (safe because n_other >= 1 in this dataset)
    df['RelSizeRatio'] = df['n_focal'] / df['n_other']

    # Sex composition differences
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Contest location: focal has home-range advantage when focal is closer to its home center
    # than the other group is to its home center (i.e., contest is relatively in focal's area)
    df['FocalHomeAdv'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Ensure binary outcome is numeric 0/1
    df['win'] = pd.to_numeric(df['win'], errors='coerce').astype('float')

    # Drop rows with missing values in any of the columns used in the model
    needed_cols = ['win', 'RelSizeDiff', 'RelSizeRatio', 'FocalHomeAdv', 'm_diff', 'f_diff', 'dyad']
    df = df.dropna(subset=needed_cols)

    # Ensure dyad is integer (useful for clustering/grouping)
    df['dyad'] = df['dyad'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting the probability that the focal
    group wins an intergroup contest as a function of relative group size, contest
    location (home-range advantage), their interaction, and control variables.

    The model estimated:
      win ~ RelSizeDiff * FocalHomeAdv + m_diff + f_diff + RelSizeRatio

    We compute cluster-robust standard errors clustered by dyad to account for
    non-independence of contests within the same dyad.

    Returns the model results object with cluster-robust covariances applied.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Formula: main effects and interaction between relative size and location
    formula = 'win ~ RelSizeDiff * FocalHomeAdv + m_diff + f_diff + RelSizeRatio'

    # Fit binomial GLM
    glm_res = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute cluster-robust covariance by dyad (allows for correlation within dyad)
    # If dyad has too few clusters, the clustered SE will be imprecise; still we provide it.
    try:
        robust_res = glm_res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If cluster adjustment fails for any reason, fall back to the default result
        robust_res = glm_res

    # Return the robust results object (or glm_res if clustering failed)
    return robust_res


