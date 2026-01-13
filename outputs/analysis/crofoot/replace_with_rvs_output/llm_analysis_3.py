from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import norm

import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_with_rvs_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw capuchin contest data into variables used in the model.

    Produces the following new columns (used later by the model):
      - rel_size_z: standardized (z) difference in group size (n_focal - n_other)
      - focal_central: binary indicator (1 if focal is closer to its home-range center than other, else 0)
      - rel_size_x_central: interaction rel_size_z * focal_central
      - male_diff_z: standardized difference in number of males (m_focal - m_other)
      - total_size_z: standardized total group size (n_focal + n_other)
      - win: retained outcome (0/1)
      - dyad: retained for clustering
    """
    # work on a copy
    df = df.copy()

    # Drop rows with missing values in columns required to compute predictors / outcome
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    df = df.dropna(subset=required_cols)

    # Relative size (focal - other)
    df['rel_size'] = df['n_focal'] - df['n_other']

    # Total contest size
    df['total_size'] = df['n_focal'] + df['n_other']

    # Male difference
    df['male_diff'] = df['m_focal'] - df['m_other']

    # Location advantage: focal is closer to its home-range center than the other group
    # (smaller distance -> more central)
    df['focal_central'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize numeric predictors to z-scores (robust to constant columns)
    def zscore(s: pd.Series) -> pd.Series:
        mean = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return s - mean
        return (s - mean) / std

    df['rel_size_z'] = zscore(df['rel_size'])
    df['male_diff_z'] = zscore(df['male_diff'])
    df['total_size_z'] = zscore(df['total_size'])

    # Interaction between relative size and focal centrality (moderation term)
    df['rel_size_x_central'] = df['rel_size_z'] * df['focal_central']

    # Ensure dyad is integer (used for clustering in modeling)
    # Keep as is if already integer-like; coerce safely
    df['dyad'] = df['dyad'].astype(int)

    # Keep only columns needed for modeling plus original outcome and dyad
    keep_cols = ['win', 'dyad', 'rel_size_z', 'focal_central', 'rel_size_x_central', 'male_diff_z', 'total_size_z']
    df = df.loc[:, keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting probability that the focal group wins.

    Model: win ~ rel_size_z + focal_central + rel_size_x_central + male_diff_z + total_size_z
    Robust (clustered) standard errors are computed with dyad as the cluster variable to account
    for non-independence of interactions between the same pair of groups.

    Returns an object that exposes params, cov_params (clustered), bse, conf_int(), and summary().
    """
    # Ensure required columns are present
    required = ['win', 'dyad', 'rel_size_z', 'focal_central', 'rel_size_x_central', 'male_diff_z', 'total_size_z']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare design matrix
    X = df[['rel_size_z', 'focal_central', 'rel_size_x_central', 'male_diff_z', 'total_size_z']].copy()
    # Ensure dtype numeric
    X['focal_central'] = X['focal_central'].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['win'].astype(float)

    # Fit binomial GLM
    glm_mod = sm.GLM(y, X, family=sm.families.Binomial())
    glm_res = glm_mod.fit()

    # Obtain cluster-robust covariance (clustered by dyad)
    # statsmodels' GLMResults may not provide get_robustcov_results in all versions;
    # compute clustered covariance matrix directly and wrap results.
    clusters = df['dyad'].values
    cov = cov_cluster(glm_res, clusters)

    class ClusteredResults:
        def __init__(self, base_res, cov_matrix):
            self._base = base_res
            self.params = base_res.params
            self.cov_params = cov_matrix
            self.bse = np.sqrt(np.diag(cov_matrix))

        def conf_int(self, alpha=0.05):
            z = norm.ppf(1 - alpha / 2)
            lower = self.params - z * self.bse
            upper = self.params + z * self.bse
            return pd.DataFrame({0: lower, 1: upper}, index=self.params.index)

        def summary(self):
            # Return the original summary object; it reflects model fit.
            # Note: the displayed standard errors in this object are the model-based ones.
            # Users can inspect bse / cov_params for clustered estimates.
            return self._base.summary()

        def __getattr__(self, item):
            # Delegate attribute access to the base results where appropriate
            return getattr(self._base, item)

    results_cluster = ClusteredResults(glm_res, cov)

    # Print summary (including clustered SEs note)
    try:
        print(results_cluster.summary())
    except Exception:
        # If summary printing fails for any reason, skip
        pass

    # Also compute and show odds ratios with 95% CIs for interpretation using clustered cov
    try:
        params = results_cluster.params
        conf = results_cluster.conf_int()
        or_df = pd.DataFrame({
            'OR': np.exp(params),
            'CI_lower': np.exp(conf[0]),
            'CI_upper': np.exp(conf[1])
        })
        print('\nOdds ratios and 95% CIs (clustered SEs):')
        print(or_df)
    except Exception:
        # if something fails, skip odds ratios
        pass

    return results_cluster