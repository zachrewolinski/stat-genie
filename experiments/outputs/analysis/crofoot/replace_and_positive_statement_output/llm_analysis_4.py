from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import scipy.stats as scistats
import matplotlib.pyplot as plt
import pickle

# Optional: read data (this line can be commented out when importing this module elsewhere)
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_and_positive_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin contest dataframe into a modeling dataframe.

    Creates the following new columns used in the model:
      - SizeDiff: n_focal - n_other
      - LocAdv: dist_other - dist_focal
      - m_diff: m_focal - m_other
      - f_diff: f_focal - f_other
      - z-scored versions: SizeDiff_z, LocAdv_z, m_diff_z, f_diff_z

    Also drops rows with missing values for any of the required columns.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'win', 'dist_focal', 'dist_other',
        'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Compute raw predictors
    df['SizeDiff'] = df['n_focal'] - df['n_other']
    # Also keep ratio for diagnostics (not used in primary model)
    # Avoid division by zero
    df['SizeRatio'] = df['n_focal'] / df['n_other'].replace(0, np.nan)

    # Location advantage: positive => contest location favors focal group
    df['LocAdv'] = df['dist_other'] - df['dist_focal']

    # Composition differences
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Standardize continuous predictors (z-score) to aid interpretation/stability
    # Use ddof=0 to use population std (numpy default)
    def zscore(series: pd.Series) -> pd.Series:
        return (series - series.mean()) / series.std(ddof=0)

    df['SizeDiff_z'] = zscore(df['SizeDiff'])
    df['LocAdv_z'] = zscore(df['LocAdv'])
    df['m_diff_z'] = zscore(df['m_diff'])
    df['f_diff_z'] = zscore(df['f_diff'])

    # Ensure dyad is integer or categorical (keep original values for clustering)
    # Try to coerce to integer if possible, otherwise keep as object
    try:
        df['dyad'] = df['dyad'].astype(int)
    except Exception:
        df['dyad'] = df['dyad'].astype(str)

    # Final dataframe returned contains original columns plus derived ones.
    # Columns needed for modeling: 'win', 'SizeDiff_z', 'LocAdv_z', 'm_diff_z', 'f_diff_z', 'dyad'
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression to test whether relative group size and contest location
    influence the probability that the focal group wins an intergroup contest.

    Model specification:
      win ~ SizeDiff_z * LocAdv_z + m_diff_z + f_diff_z

    We fit a GLM with binomial family and compute cluster-robust standard errors clustered on 'dyad'
    to account for non-independence of contests within the same dyad.

    Returns a dictionary containing the fitted model result and cluster-robust covariance/SEs.
    """
    # Ensure required columns are present
    needed = ['win', 'SizeDiff_z', 'LocAdv_z', 'm_diff_z', 'f_diff_z', 'dyad']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit GLM (logistic regression)
    formula = 'win ~ SizeDiff_z * LocAdv_z + m_diff_z + f_diff_z'
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = glm_model.fit()

    # Compute cluster-robust covariance matrix clustered by dyad using sandwich estimator
    # cov_cluster returns a covariance matrix aligned with model parameters
    try:
        cluster_groups = df['dyad'].values
        clustered_cov = cov_cluster(res, cluster_groups)
        clustered_se = np.sqrt(np.diag(clustered_cov))
        # Compute robust z-scores and p-values (Wald-type using normal approximation)
        params = res.params.values
        z_robust = params / clustered_se
        pvalues_robust = 2 * (1 - scistats.norm.cdf(np.abs(z_robust)))
    except Exception:
        # If cluster covariance fails, set cluster outputs to None
        clustered_cov = None
        clustered_se = None
        pvalues_robust = None

    # For convenience, also attempt to compute average marginal effects (AME) from the original result
    try:
        marg = res.get_margeff(at='overall', method='dydx')
    except Exception:
        marg = None

    # Return a dictionary with key outputs
    results = {
        'model_result': res,                    # original fitted GLM results
        'clustered_cov': clustered_cov,         # clustered covariance matrix (or None)
        'clustered_se': clustered_se,           # clustered standard errors (or None)
        'clustered_pvalues': pvalues_robust,    # p-values based on clustered SEs (or None)
        'aic': getattr(res, 'aic', None),
        'bic': getattr(res, 'bic', None),
        'marginal_effects': marg
    }

    return results