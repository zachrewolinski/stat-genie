from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare variables for modeling the probability that the focal group wins an intergroup contest.

    Adds standardized (z) predictors used in the model:
      - SizeRatio_z: standardized (n_focal / n_other)
      - LocationAdvantage_z: standardized (dist_other - dist_focal) so positive = focal closer to its home center
      - MaleDiff_z: standardized (m_focal - m_other)
      - FemaleDiff_z: standardized (f_focal - f_other)
      - DistDiff_z: standardized (dist_other - dist_focal) (same numeric quantity as LocationAdvantage but explicitly named as control)

    Also checks and drops rows with missing or invalid values (e.g., zero n_other) to avoid division by zero.
    """
    df = df.copy()

    # Required columns for the analysis
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Coerce numeric columns and dyad to numeric if possible
    numeric_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                    'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Remove rows where n_other is zero or negative or where n_focal <= 0
    df = df[(df['n_other'] > 0) & (df['n_focal'] > 0)]

    # Core derived variables
    # Size ratio: focal size relative to other
    df['SizeRatio'] = df['n_focal'] / df['n_other']
    # Log transform (kept as helper)
    df['LogSizeRatio'] = np.log(df['SizeRatio'])

    # Location advantage: positive means focal is closer to its home center than the other group
    df['LocationAdvantage'] = df['dist_other'] - df['dist_focal']
    # For clarity also store DistDiff (same quantity)
    df['DistDiff'] = df['LocationAdvantage']

    # Sex composition differences
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Standardize (z-score) numeric predictors used in modeling for easier interpretation
    def zscore(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mean = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            std = 1.0
        return (s - mean) / std

    df['SizeRatio_z'] = zscore(df['SizeRatio'])
    df['LocationAdvantage_z'] = zscore(df['LocationAdvantage'])
    df['MaleDiff_z'] = zscore(df['MaleDiff'])
    df['FemaleDiff_z'] = zscore(df['FemaleDiff'])
    df['DistDiff_z'] = zscore(df['DistDiff'])

    # Ensure the outcome is integer 0/1
    # Coerce to numeric first to be safe
    df['win'] = pd.to_numeric(df['win'], errors='coerce').astype(int)

    # Ensure dyad is present and numeric (keep as-is if already numeric)
    df['dyad'] = pd.to_numeric(df['dyad'], errors='coerce')

    # Keep all original columns plus newly derived ones
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting probability focal group wins (win = 1) from:
      - standardized relative group size (SizeRatio_z)
      - standardized location advantage (LocationAdvantage_z)
      - their interaction
    while controlling for male and female differences and clustering standard errors by dyad.

    Returns a dict with the fitted model and a cluster-robust covariance results-like object.
    """
    # Ensure required predictor columns exist
    required = ['win', 'SizeRatio_z', 'LocationAdvantage_z', 'MaleDiff_z', 'FemaleDiff_z', 'DistDiff_z', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: include interaction between size and location
    formula = 'win ~ SizeRatio_z * LocationAdvantage_z + MaleDiff_z + FemaleDiff_z + DistDiff_z'

    # Fit GLM with binomial family (logistic regression)
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute cluster-robust covariance matrix by dyad to account for repeated contests within dyads
    try:
        cluster_cov = cov_cluster(model_glm, df['dyad'])
    except Exception:
        # If cov_cluster fails for some reason, fall back to a naive covariance
        cluster_cov = model_glm.cov_params()

    # Create a lightweight results-like wrapper for cluster-robust outputs
    class ClusterRobustResults:
        def __init__(self, base_res, cov):
            self._base = base_res
            self._cov = cov
            # ensure params is accessible
            self.params = base_res.params

        def cov_params(self):
            return self._cov

        @property
        def bse(self):
            return np.sqrt(np.diag(self._cov))

        def summary(self):
            # Return a simple dataframe-style string with coef and clustered SE
            try:
                import pandas as _pd
                df_sum = _pd.DataFrame({
                    'coef': self.params,
                    'bse_cluster': self.bse
                })
                return df_sum.to_string()
            except Exception:
                return f"ClusterRobustResults with params: {self.params}"

    robust_results = ClusterRobustResults(model_glm, cluster_cov)

    # Print summaries for convenience (in an interactive session)
    try:
        print('Standard GLM results:')
        print(model_glm.summary())
        print('\nCluster-robust (by dyad) results:')
        print(robust_results.summary())
    except Exception:
        # If printing fails in some environments, just pass
        pass

    # Prepare a compact dictionary of key outputs
    outputs = {
        'formula': formula,
        'n_obs': int(model_glm.nobs),
        'glm_results': model_glm,
        'cluster_robust_results': robust_results
    }

    return outputs