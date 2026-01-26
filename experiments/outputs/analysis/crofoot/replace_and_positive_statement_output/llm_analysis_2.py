from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import scipy  # noqa: F401
import statsmodels.api as sm
import statsmodels.formula.api as smf  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import pickle  # noqa: F401

# Example read (left in place from original file; can be removed or modified as needed)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_and_positive_statement_output/crofoot.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare variables needed for modelling intergroup contest outcomes.
    Produces these new columns (kept in the returned dataframe):
      - log_size_ratio: log(n_focal / n_other)
      - log_size_ratio_c: mean-centered log_size_ratio (used in model)
      - focal_home: 1 if dist_focal < dist_other else 0
      - dist_diff: dist_other - dist_focal (positive means focal is closer)
      - male_diff: m_focal - m_other
      - female_diff: f_focal - f_other
    Also drops rows with missing values for core variables.

    Returns a dataframe containing at least the columns required by the model:
      ['win', 'dyad', 'log_size_ratio_c', 'focal_home',
       'dist_diff', 'male_diff', 'female_diff']
    """
    df = df.copy()

    # Required input columns
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing values in any of the required columns
    df = df.dropna(subset=required)

    # Ensure numeric types where appropriate
    df['n_focal'] = pd.to_numeric(df['n_focal'], errors='coerce')
    df['n_other'] = pd.to_numeric(df['n_other'], errors='coerce')
    df['dist_focal'] = pd.to_numeric(df['dist_focal'], errors='coerce')
    df['dist_other'] = pd.to_numeric(df['dist_other'], errors='coerce')
    df['m_focal'] = pd.to_numeric(df['m_focal'], errors='coerce')
    df['m_other'] = pd.to_numeric(df['m_other'], errors='coerce')
    df['f_focal'] = pd.to_numeric(df['f_focal'], errors='coerce')
    df['f_other'] = pd.to_numeric(df['f_other'], errors='coerce')
    df['win'] = pd.to_numeric(df['win'], errors='coerce')
    # Re-drop any rows that became NA after coercion
    df = df.dropna(subset=required)

    # Relative size: log ratio (assume n_* > 0)
    # To be safe, avoid division by zero or non-positive sizes
    positive_mask = (df['n_focal'] > 0) & (df['n_other'] > 0)
    if not positive_mask.all():
        df = df.loc[positive_mask].copy()

    df['log_size_ratio'] = np.log(df['n_focal'] / df['n_other'])

    # Center the log ratio to improve interpretability of main effects
    df['log_size_ratio_c'] = df['log_size_ratio'] - df['log_size_ratio'].mean()

    # Location advantage: focal is closer to its home-range center than other
    df['focal_home'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Continuous distance difference (positive => focal closer)
    df['dist_diff'] = df['dist_other'] - df['dist_focal']

    # Composition differences
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Keep the columns important for modelling and inference (may include helper columns)
    keep_cols = ['win', 'dyad', 'log_size_ratio', 'log_size_ratio_c',
                 'focal_home', 'dist_diff', 'male_diff', 'female_diff']

    # Reset index for cleanliness
    return df[keep_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial) predicting the probability that the focal group wins.
    Primary predictors: log_size_ratio_c, focal_home. Controls: dist_diff, male_diff, female_diff.
    Clustered (dyad) robust standard errors are computed and returned in the results wrapper.
    """
    # Copy to avoid modifying original
    df = df.copy()

    # Ensure required columns are present
    required = ['win', 'dyad', 'log_size_ratio_c', 'focal_home',
                'dist_diff', 'male_diff', 'female_diff']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Design matrix: only the conceptual variables (plus intercept)
    X = df[['log_size_ratio_c', 'focal_home', 'dist_diff', 'male_diff', 'female_diff']].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['win'].astype(float)

    # Fit logistic regression via Logit (provides get_robustcov_results for clustered SEs)
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)

    # Obtain cluster-robust (dyad) covariance
    # Use the results method to get a results instance with adjusted covariance
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except AttributeError:
        # As a fallback (if get_robustcov_results is not available), compute clustered covariance manually
        # and attach it to the results via the get_robustcov_results API when possible.
        from statsmodels.stats.sandwich_covariance import cov_cluster
        cov = cov_cluster(res, df['dyad'])
        # Create a results wrapper-like object by copying res and overriding cov_params and bse
        # Note: This is a lightweight fallback; prefer the primary approach above.
        class _ResWithCov:
            def __init__(self, res, cov):
                self._res = res
                self.cov_params_default = cov

            def __getattr__(self, item):
                return getattr(self._res, item)

            def cov_params(self):
                return self.cov_params_default

            @property
            def bse(self):
                return np.sqrt(np.diag(self.cov_params_default))

            def summary(self, *args, **kwargs):
                return self._res.summary(*args, **kwargs)

        res_clust = _ResWithCov(res, cov)

    # Print a concise summary and return the robust results object for programmatic use
    print(res_clust.summary())
    return res_clust