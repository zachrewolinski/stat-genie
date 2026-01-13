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
    Transform raw contest data into variables required for the binomial model.

    Produces the following model columns (exact names used in the modeling function):
      - win: binary outcome (kept from input)
      - SizeDiff: raw difference n_focal - n_other
      - LocAdv: raw location advantage dist_other - dist_focal
      - MaleDiff: m_focal - m_other
      - FemaleDiff: f_focal - f_other
      - SizeDiff_c, LocAdv_c, MaleDiff_c, FemaleDiff_c: mean-centered versions
      - dyad: dyad identifier (kept as-is for fixed effects and clustering)
    """

    # Work on a copy
    df = df.copy()

    # Required columns
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with missing values in any of the required columns
    df = df.dropna(subset=required_cols)

    # Create raw difference variables
    df['SizeDiff'] = df['n_focal'] - df['n_other']
    # Alternative scale (ratio) could be used; here we use difference because group sizes are small integers
    df['LocAdv'] = df['dist_other'] - df['dist_focal']
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Mean-center continuous predictors for interpretability and to reduce collinearity in interactions
    for col in ['SizeDiff', 'LocAdv', 'MaleDiff', 'FemaleDiff']:
        mean_val = df[col].mean()
        df[col + '_c'] = df[col] - mean_val

    # Ensure dyad is present and keep it for fixed effects and clustering
    # Make dyad a categorical-like column but keep original values for clustering
    df['dyad'] = df['dyad'].astype(int)

    # Keep only the columns necessary for modeling (but return full df to be safe)
    # Explicitly ensure the columns named in the conceptual variables exist
    expected_model_cols = ['win', 'SizeDiff_c', 'LocAdv_c', 'MaleDiff_c', 'FemaleDiff_c', 'dyad',
                           'SizeDiff', 'LocAdv', 'MaleDiff', 'FemaleDiff']
    for c in expected_model_cols:
        if c not in df.columns:
            raise RuntimeError(f"Expected column '{c}' not found after transformation")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic (binomial) regression testing the effects of relative group size (SizeDiff_c),
    location advantage (LocAdv_c), and their interaction on the probability that the focal group wins.

    The model includes MaleDiff_c and FemaleDiff_c as controls and dyad fixed effects.
    We also compute cluster-robust standard errors clustered by dyad.

    Returns a dictionary with the fitted GLM object and cluster-robust results.
    """

    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['win', 'SizeDiff_c', 'LocAdv_c', 'MaleDiff_c', 'FemaleDiff_c', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe missing required columns for modeling: {missing}")

    # Fit binomial GLM with dyad fixed effects; use interaction operator * to include main effects and interaction
    formula = 'win ~ SizeDiff_c * LocAdv_c + MaleDiff_c + FemaleDiff_c + C(dyad)'
    glm_res = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute cluster-robust covariance by dyad
    try:
        robust_res = glm_res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If cluster robust fails for any reason, return the original glm result and note that robust fit failed
        robust_res = None

    # Return both the original and robust results (robust_res may be None if computation failed)
    return {
        'glm_result': glm_res,
        'cluster_robust_result': robust_res
    }


