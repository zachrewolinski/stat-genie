from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Required columns for the analysis
    required = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required)

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Compute relative size as log ratio (log(n_focal / n_other)). Use a small eps if needed (not necessary here given min values >0)
    df['rel_size_log'] = np.log(df['n_focal'] / df['n_other'])

    # Compute location advantage for focal: positive if contest is closer to focal home center than other group's center
    df['location_adv'] = df['dist_other'] - df['dist_focal']

    # Sex composition differences (focal - other)
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Standardize continuous predictors (z-score). Use ddof=0 for population-style std.
    df['rel_size_z'] = (df['rel_size_log'] - df['rel_size_log'].mean()) / df['rel_size_log'].std(ddof=0)
    df['location_adv_z'] = (df['location_adv'] - df['location_adv'].mean()) / df['location_adv'].std(ddof=0)
    df['m_diff_z'] = (df['m_diff'] - df['m_diff'].mean()) / df['m_diff'].std(ddof=0)
    df['f_diff_z'] = (df['f_diff'] - df['f_diff'].mean()) / df['f_diff'].std(ddof=0)

    # Interaction term between relative size and location advantage to test whether location moderates the size effect
    df['rel_size_x_location'] = df['rel_size_z'] * df['location_adv_z']

    # Ensure dyad is integer (for clustering later)
    df['dyad'] = df['dyad'].astype(int)

    # Keep only the columns necessary for modeling and diagnostics
    keep_cols = ['win', 'rel_size_z', 'location_adv_z', 'rel_size_x_location', 'm_diff_z', 'f_diff_z', 'dyad',
                 # keep raw versions for possible checks
                 'rel_size_log', 'location_adv', 'm_diff', 'f_diff', 'n_focal', 'n_other', 'dist_focal', 'dist_other']
    # Some of these may not exist if earlier drops removed rows; select intersection
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Prepare predictors and outcome
    predictors = ['rel_size_z', 'location_adv_z', 'rel_size_x_location', 'm_diff_z', 'f_diff_z']
    # Ensure predictors exist in df
    missing = [p for p in predictors if p not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing predictors in dataframe: {missing}")

    X = sm.add_constant(df[predictors])
    y = df['win']

    # Fit logistic regression (binomial) using maximum likelihood
    logit_res = sm.Logit(y, X).fit(disp=False)

    # Obtain cluster-robust standard errors clustered by dyad to account for non-independence
    try:
        # Attempt to compute clustered covariance matrix
        clustered_cov = cov_cluster(logit_res, df['dyad'])
    except Exception:
        # If clustering fails for any reason, fall back to heteroskedasticity-robust SEs (HC1)
        clustered_cov = cov_hc1(logit_res)

    # Compute robust standard errors, t-values, p-values, and confidence intervals based on the covariance matrix
    params = logit_res.params
    bse = np.sqrt(np.diag(clustered_cov))
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tvalues = params / bse
    pvalues = 2 * (1 - norm.cdf(np.abs(tvalues)))

    def conf_int(alpha: float = 0.05) -> np.ndarray:
        z = norm.ppf(1 - alpha / 2)
        lower = params - z * bse
        upper = params + z * bse
        return np.column_stack((lower, upper))

    # Attach/override attributes on the result object so downstream code using .summary(), .bse, .pvalues, etc. will see robust estimates
    # Many statsmodels summary methods read these attributes, so overwriting them updates the displayed results.
    setattr(logit_res, 'cov_params_default', lambda: clustered_cov)
    setattr(logit_res, 'cov_params', lambda: clustered_cov)
    setattr(logit_res, 'bse', bse)
    setattr(logit_res, 'tvalues', tvalues)
    setattr(logit_res, 'pvalues', pvalues)
    setattr(logit_res, 'conf_int', conf_int)

    return logit_res