from typing import Any, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats as scipy_stats


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/replace_with_rvs_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw contest dataset into the final dataframe used for modeling.

    Produced columns:
    - win: binary outcome (int)
    - RelSize_z: z-scored (n_focal - n_other)
    - LocAdv_z: z-scored (dist_other - dist_focal)
    - RelSize_x_LocAdv: interaction term RelSize_z * LocAdv_z
    - MaleDiff_z: z-scored (m_focal - m_other)
    - TotalSize_z: z-scored (n_focal + n_other)
    - dyad: dyad identifier (kept for clustering)
    """
    df = df.copy()

    # Ensure required columns exist and drop rows with missing values in those columns
    required = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for transform: {missing}")

    df = df.dropna(subset=required)

    # Compute raw predictors (helper columns)
    df['RelSize'] = df['n_focal'] - df['n_other']
    df['LocAdv'] = df['dist_other'] - df['dist_focal']
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Standardize (z-score) the continuous predictors, guarding against zero std
    for col in ['RelSize', 'LocAdv', 'MaleDiff', 'TotalSize']:
        mean = df[col].mean()
        std = df[col].std()
        if std is None or std == 0 or np.isnan(std):
            # fallback to zero column if no variation
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Interaction term (use the z-scored versions)
    df['RelSize_x_LocAdv'] = df['RelSize_z'] * df['LocAdv_z']

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Keep only the columns needed for modeling
    out_cols = ['win', 'RelSize_z', 'LocAdv_z', 'RelSize_x_LocAdv', 'MaleDiff_z', 'TotalSize_z', 'dyad']
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting probability focal group wins (win) from relative size,
    location advantage, their interaction, and controls. Use cluster-robust standard errors
    clustered by dyad to account for repeated observations of dyads.

    Returns:
    - results: object that behaves like a statsmodels results object but with cluster-robust
      covariance, bse, and pvalues available. The object provides a get_robustcov_results
      method for compatibility.
    """
    df = df.copy()

    # Check columns
    model_cols = ['RelSize_z', 'LocAdv_z', 'RelSize_x_LocAdv', 'MaleDiff_z', 'TotalSize_z']
    for c in model_cols + ['win', 'dyad']:
        if c not in df.columns:
            raise KeyError(f"Required column for modeling missing: {c}")

    X = df[model_cols]
    X = sm.add_constant(X)
    y = df['win']

    # Fit binomial (logistic) regression
    logit_model = sm.Logit(y, X)
    fitted = logit_model.fit(disp=False)

    # Compute cluster-robust covariance matrix clustered on dyad
    # cov_cluster accepts the fitted results object and the grouping variable
    clustered_cov = cov_cluster(fitted, df['dyad'])

    # Build a lightweight wrapper around the fitted results that exposes
    # clustered covariance, clustered standard errors, and p-values.
    class ClusteredResults:
        def __init__(self, base_results, cov):
            self._base = base_results
            # Ensure param index is available
            try:
                self.params = base_results.params
            except Exception:
                # fallback to numpy array if needed
                self.params = pd.Series(base_results.params, index=[f'param_{i}' for i in range(len(base_results.params))])

            # Convert covariance to DataFrame with appropriate labels
            self.cov_params = pd.DataFrame(cov, index=self.params.index, columns=self.params.index)
            # Standard errors from the clustered covariance
            self.bse = pd.Series(np.sqrt(np.diag(cov)), index=self.params.index)
            # z-stats and p-values (two-sided using normal approximation)
            zvals = self.params / self.bse
            self.pvalues = pd.Series(2.0 * scipy_stats.norm.sf(np.abs(zvals)), index=self.params.index)

        def get_robustcov_results(self, cov_type='cluster', groups=None):
            # Return self for compatibility; cov_type/groups are informational here.
            return self

        def summary(self, *args, **kwargs):
            # Delegate to base results summary if possible
            try:
                return self._base.summary(*args, **kwargs)
            except Exception:
                return f'ClusteredResults for model with params:\n{self.params}'

        def __getattr__(self, item):
            # Delegate attribute access to the underlying results object where possible
            if item in ('params', 'cov_params', 'bse', 'pvalues', 'get_robustcov_results', 'summary'):
                return object.__getattribute__(self, item)
            return getattr(self._base, item)

    results_clustered = ClusteredResults(fitted, clustered_cov)
    return results_clustered