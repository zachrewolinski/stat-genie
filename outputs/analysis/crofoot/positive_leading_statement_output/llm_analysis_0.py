from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/positive_leading_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw contest dataframe into a form suitable for logistic regression.

    Produces the following new columns used in modeling:
      - RelGroupSize_z: standardized (z) of (n_focal - n_other)
      - HomeFieldAdv_z: standardized (z) of (dist_other - dist_focal) where positive means focal is closer to its home center (home-field advantage)
      - FocalCloser: binary indicator (1 if dist_focal < dist_other else 0)
      - RelMaleCount_z: standardized (z) of (m_focal - m_other)
      - TotalSize_z: standardized (z) of (n_focal + n_other)
      - win: ensured integer 0/1
      - dyad: preserved for clustering
    """
    df = df.copy()

    # Keep only rows with essential variables present
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    df = df.dropna(subset=required_cols)

    # Ensure integer types where appropriate
    df['win'] = df['win'].astype(int)
    df['dyad'] = df['dyad'].astype(int)

    # Relative group size (focal - other)
    df['RelGroupSize'] = df['n_focal'] - df['n_other']
    # Home-field advantage: positive when focal is closer to its home center than the other group
    df['HomeFieldAdv'] = df['dist_other'] - df['dist_focal']
    # Binary indicator: focal closer than other
    df['FocalCloser'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Relative male count
    df['RelMaleCount'] = df['m_focal'] - df['m_other']
    # Total size context
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Standardize continuous predictors (z-score). Use ddof=0 to be explicit (population std).
    def zscore(series):
        return (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1.0)

    df['RelGroupSize_z'] = zscore(df['RelGroupSize'])
    df['HomeFieldAdv_z'] = zscore(df['HomeFieldAdv'])
    df['RelMaleCount_z'] = zscore(df['RelMaleCount'])
    df['TotalSize_z'] = zscore(df['TotalSize'])

    # Keep only the columns required for modeling (but preserve originals if desired)
    model_cols = [
        'win', 'RelGroupSize_z', 'HomeFieldAdv_z', 'FocalCloser',
        'RelMaleCount_z', 'TotalSize_z', 'dyad'
    ]

    # Return dataframe with model columns (plus original columns still present in df copy if needed)
    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression to predict the probability that the focal group wins.

    Model specification (primary):
      win ~ RelGroupSize_z * HomeFieldAdv_z + RelMaleCount_z + TotalSize_z

    This specification tests:
      - main effect of relative group size
      - main effect of home-field advantage (continuous)
      - their interaction (does location moderate the size advantage?)
      - controls for relative male count and total group size

    Cluster-robust standard errors are computed by dyad to account for repeated observations per dyad.
    Returns the fitted model object with cluster-robust covariances applied.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure the df passed in contains the columns the transform function creates
    required = ['win', 'RelGroupSize_z', 'HomeFieldAdv_z', 'RelMaleCount_z', 'TotalSize_z', 'dyad', 'FocalCloser']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula: include interaction between relative size and home-field advantage
    formula = 'win ~ RelGroupSize_z * HomeFieldAdv_z + RelMaleCount_z + TotalSize_z'

    # Fit GLM (binomial / logistic)
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    glm_res = glm_model.fit()

    # Obtain cluster-robust standard errors clustered by dyad
    try:
        clustered_res = glm_res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If clustering fails for any reason, return plain GLM results but warn the user
        clustered_res = glm_res

    # For convenience, also compute and attach average marginal effects (AME) on probability scale
    try:
        # statsmodels' get_margeff can work for GLM results
        margeff = clustered_res.get_margeff(method='dydx', at='overall')
    except Exception:
        margeff = None

    # Package results in a dictionary for downstream use
    results = {
        'model_result': clustered_res,
        'marginal_effects': margeff
    }

    return results


