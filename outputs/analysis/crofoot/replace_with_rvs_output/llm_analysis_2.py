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
    Transform the raw dataset to create the variables used in the model.

    Produces the following new columns required by the model:
      - log_size_ratio_z: z-scored log((n_focal+0.1)/(n_other+0.1))
      - ProximityAdvantage_z: z-scored (dist_other - dist_focal)
      - male_diff_z, female_diff_z, total_n_z: z-scored control differences/totals
      - size_x_location: interaction term (log_size_ratio_z * ProximityAdvantage_z)
      - InFocalHome: binary indicator (1 if contest closer to focal home center)

    Returns the transformed dataframe (rows with missing required vars are dropped).
    """
    df = df.copy()

    # Ensure numeric types for key columns (coerce errors to NaN)
    numeric_cols = ['n_focal', 'n_other', 'dist_focal', 'dist_other',
                    'm_focal', 'm_other', 'f_focal', 'f_other', 'win', 'dyad']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the required raw inputs for our constructed variables
    required_raw = ['n_focal', 'n_other', 'dist_focal', 'dist_other', 'win', 'dyad']
    df = df.dropna(subset=required_raw)

    # Relative group size: stabilized log ratio (avoid division by zero)
    df['log_size_ratio'] = np.log((df['n_focal'] + 0.1) / (df['n_other'] + 0.1))
    df['size_diff'] = df['n_focal'] - df['n_other']

    # Contest location: positive => contest closer to focal home center
    df['ProximityAdvantage'] = df['dist_other'] - df['dist_focal']
    df['InFocalHome'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Composition controls
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']
    df['total_n'] = df['n_focal'] + df['n_other']

    # Z-score the continuous predictors for interpretability
    to_z = ['log_size_ratio', 'ProximityAdvantage', 'male_diff', 'female_diff', 'total_n']
    for col in to_z:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # If std is zero (unlikely here), fill with zeros to avoid division by zero
        if pd.isna(std) or std == 0:
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Interaction between relative size and location to test whether the size advantage depends on location
    df['size_x_location'] = df['log_size_ratio_z'] * df['ProximityAdvantage_z']

    # Keep only columns needed for modeling plus identifiers for reference
    needed = ['win', 'dyad', 'log_size_ratio_z', 'ProximityAdvantage_z',
              'size_x_location', 'male_diff_z', 'female_diff_z', 'total_n_z',
              'log_size_ratio', 'ProximityAdvantage', 'male_diff', 'female_diff', 'total_n',
              'InFocalHome', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
              'm_focal', 'm_other', 'f_focal', 'f_other', 'focal', 'other', 'dyad']

    # Keep intersection of needed and actual columns
    needed_present = [c for c in needed if c in df.columns]
    df = df[needed_present].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic (binomial) regression predicting the probability that the focal group wins.

    Model specification (primary):
      win ~ log_size_ratio_z + ProximityAdvantage_z + size_x_location
            + male_diff_z + female_diff_z + total_n_z

    We cluster-robust the standard errors by dyad to account for repeated observations of the same pair.

    Returns: the fitted results object with cluster-robust SEs (if available).
    """
    import statsmodels.api as sm

    df = df.copy()

    # Required predictor names (these must be present in the transformed dataframe)
    predictors = ['log_size_ratio_z', 'ProximityAdvantage_z', 'size_x_location',
                  'male_diff_z', 'female_diff_z', 'total_n_z']

    # Drop rows with missing data in predictors/response/cluster
    df_model = df.dropna(subset=predictors + ['win', 'dyad'])

    # Prepare design matrices
    X = sm.add_constant(df_model[predictors])
    y = df_model['win']

    # Fit GLM (binomial / logistic)
    glm_res = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Obtain cluster-robust covariance (clustered by dyad) where possible
    try:
        res_cluster = glm_res.get_robustcov_results(cov_type='cluster', groups=df_model['dyad'])
    except Exception:
        # If clustering fails for any reason, return the plain GLM results
        res_cluster = glm_res

    # For convenience, also compute and attach average marginal effects at means
    try:
        marg_eff = res_cluster.get_margeff(method='dydx', at='mean')
        res_cluster.margeff = marg_eff
    except Exception:
        # not critical; ignore if marginal effects cannot be computed
        res_cluster.margeff = None

    return res_cluster


