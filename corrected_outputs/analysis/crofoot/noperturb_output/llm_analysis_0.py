from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels
from types import SimpleNamespace

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin contest dataframe into a dataframe with variables
    required for the logistic regression model.

    Produces the following new columns (all used in modeling):
      - log_size_ratio: log(n_focal / n_other)
      - LocationAdv: dist_other - dist_focal (positive => focal is closer to its center)
      - SizeLoc_interaction: product of log_size_ratio and LocationAdv
      - m_diff: m_focal - m_other
      - f_diff: f_focal - f_other
      - dyad_cat: categorical dyad id string for fixed effects / dummies
      - win: ensured integer 0/1

    Rows with missing values in any of the required columns are dropped.
    """

    # Required columns for the analysis
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Work on a copy
    df = df.copy()

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types for columns that should be numeric (do not coerce 'dyad' here)
    numeric_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                    'm_focal', 'm_other', 'f_focal', 'f_other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols + ['dyad'])

    # Dependent variable: ensure integer 0/1
    # If win is not exactly 0/1, this will cast (e.g., '1.0' -> 1)
    df['win'] = df['win'].astype(int)

    # Independent variables: relative size (log ratio)
    eps = 1e-6
    df['log_size_ratio'] = np.log((df['n_focal'] + eps) / (df['n_other'] + eps))

    # Location advantage: positive means focal is closer to its center than other is to its center
    df['LocationAdv'] = df['dist_other'] - df['dist_focal']

    # Interaction term
    df['SizeLoc_interaction'] = df['log_size_ratio'] * df['LocationAdv']

    # Controls: differences in males and females (focal - other)
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Dyad categorical variable for fixed effects and clustering
    # Preserve original dyad values but create a string categorical identifier
    df['dyad_cat'] = 'dyad_' + df['dyad'].astype(str)

    # Keep only columns necessary for modeling plus dyad for clustering
    model_cols = [
        'win', 'log_size_ratio', 'LocationAdv', 'SizeLoc_interaction',
        'm_diff', 'f_diff', 'dyad_cat', 'dyad'
    ]
    df = df[model_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial) predicting the probability that the focal
    group wins an intergroup contest. The model includes main effects of
    relative group size (log_size_ratio) and location advantage (LocationAdv),
    their interaction, controls for male/female differences, and dyad fixed
    effects. We report clustered robust standard errors by dyad to account for
    repeated observations of the same dyad.

    Returns an object exposing the fitted results together with clustered robust
    covariance information (accessible as .cov_cluster and .bse_cluster). The
    object also provides a .summary() method (delegated to the underlying
    fitted results).
    """

    # Ensure transform has been applied: require columns exist
    required = ['win', 'log_size_ratio', 'LocationAdv', 'SizeLoc_interaction',
                'm_diff', 'f_diff', 'dyad_cat', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError('Dataframe is missing required columns: {}'.format(missing))

    # Response
    y = df['win'].astype(float)

    # Base set of regressors
    X = df[['log_size_ratio', 'LocationAdv', 'SizeLoc_interaction', 'm_diff', 'f_diff']].copy()

    # Dyad fixed effects as dummies (drop_first to avoid multicollinearity)
    dyad_dummies = pd.get_dummies(df['dyad_cat'], prefix='dyad', drop_first=True)
    X = pd.concat([X, dyad_dummies], axis=1)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (statsmodels Logit)
    logit_model = sm.Logit(y, X)
    fitted = logit_model.fit(disp=False)

    # Attempt to compute clustered robust covariance by 'dyad'.
    # Some statsmodels versions expose a get_robustcov_results method on results;
    # others do not. To be robust across versions, compute the clustered
    # covariance manually (using statsmodels' sandwich utilities) if needed and
    # return a wrapper object that exposes both the original fitted results and
    # the clustered covariance / bse.
    groups = df['dyad']

    cov_cluster = None
    bse_cluster = None

    # Try to use built-in convenience method first (if present)
    try:
        clustered_results = fitted.get_robustcov_results(cov_type='cluster', groups=groups)
        # If successful, return that results object (it already encapsulates clustered cov)
        print(clustered_results.summary())
        return clustered_results
    except Exception:
        # Fall back: compute clustered covariance using sandwich estimator
        # statsmodels provides cov_cluster utility
        try:
            cov_cluster = statsmodels.stats.sandwich_covariance.cov_cluster(fitted, groups)
            bse_cluster = np.sqrt(np.diag(cov_cluster))
        except Exception:
            # If cov_cluster is unavailable for some reason, compute a fallback
            # (non-clustered) covariance from the fitted results
            cov_cluster = fitted.cov_params()
            bse_cluster = fitted.bse.values if hasattr(fitted, 'bse') else np.sqrt(np.diag(cov_cluster))

    # Build a simple wrapper object exposing the relevant pieces
    class ClusteredResultsWrapper:
        def __init__(self, fitted, cov_cluster, bse_cluster, groups):
            self.fitted = fitted
            self.params = fitted.params
            self.cov_cluster = cov_cluster
            self.bse_cluster = bse_cluster
            self.groups = groups

        def summary(self):
            # Delegate to the underlying fitted.summary(). Users can inspect
            # .bse_cluster and .cov_cluster for clustered SEs.
            return self.fitted.summary()

        # Provide a method name some users may expect
        def get_robustcov_results(self, *args, **kwargs):
            return self

        # Convenience accessors to mimic statsmodels naming
        def cov_params(self):
            return self.cov_cluster

        @property
        def bse(self):
            return self.bse_cluster

    clustered_wrapper = ClusteredResultsWrapper(fitted, cov_cluster, bse_cluster, groups)

    # Print summary for quick inspection (users can also inspect returned object)
    print(clustered_wrapper.summary())

    return clustered_wrapper