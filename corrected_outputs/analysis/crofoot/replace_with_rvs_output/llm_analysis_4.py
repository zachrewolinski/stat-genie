from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/replace_with_rvs_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling. Creates standardized predictors for relative group size,
    distance advantage (location), and male difference. Drops rows with missing data in
    the variables required for the model.

    Final dataframe will contain (at minimum) the columns:
      - win (0/1)
      - RelSize_z
      - DistAdv_z
      - MaleDiff_z
      - dyad
    """
    df = df.copy()

    # Required raw columns
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Ensure win is integer 0/1
    # If win is boolean or floats like 0.0/1.0, cast to int
    # If win contains other values, this will raise; that's appropriate.
    df['win'] = df['win'].astype(int)

    # Compute raw predictors (internal helper names) and then standardize to required final column names
    # RelSize: n_focal - n_other -> RelSize_z
    df['_RelSize_raw'] = df['n_focal'] - df['n_other']
    # DistAdv: dist_other - dist_focal -> DistAdv_z
    df['_DistAdv_raw'] = df['dist_other'] - df['dist_focal']
    # MaleDiff: m_focal - m_other -> MaleDiff_z
    df['_MaleDiff_raw'] = df['m_focal'] - df['m_other']

    # Standardize (z-score) each predictor; guard against zero std
    raw_to_final = {
        '_RelSize_raw': 'RelSize_z',
        '_DistAdv_raw': 'DistAdv_z',
        '_MaleDiff_raw': 'MaleDiff_z'
    }

    for raw_col, final_col in raw_to_final.items():
        mean = df[raw_col].mean()
        std = df[raw_col].std(ddof=0)
        if pd.isna(std) or std == 0:
            # if no variation, create a zero column to avoid dividing by zero
            df[final_col] = 0.0
        else:
            df[final_col] = (df[raw_col] - mean) / std

    # Drop internal helper raw columns
    df = df.drop(columns=[c for c in raw_to_final.keys()])

    # Return dataframe with new columns added
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability that the focal group wins an intergroup contest.

    Model specification:
      win ~ RelSize_z * DistAdv_z + MaleDiff_z

    Interaction term tests whether the effect of relative group size depends on the location advantage.

    Standard errors are clustered by 'dyad' to account for non-independence of contests within dyads.

    Returns a statsmodels results object with cluster-robust covariance (if supported by the installed statsmodels version).
    """
    # Ensure needed columns are present
    model_cols = ['win', 'RelSize_z', 'DistAdv_z', 'MaleDiff_z', 'dyad']
    missing = [c for c in model_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Fit logistic regression
    formula = 'win ~ RelSize_z * DistAdv_z + MaleDiff_z'
    logit_model = smf.logit(formula=formula, data=df)

    # Try to request cluster-robust covariance at fit time. Many statsmodels versions accept
    # cov_type and cov_kwds in fit(); if not supported, fall back to the plain fit and then
    # compute robust covariances via available utilities.
    try:
        fit_res = logit_model.fit(disp=False, cov_type='cluster', cov_kwds={'groups': df['dyad']})
        clustered_res = fit_res
    except TypeError:
        # Older/newer versions might not accept cov_type in fit; do a plain fit and then attempt
        # to attach cluster-robust covariance using statsmodels utilities.
        fit_res = logit_model.fit(disp=False)
        try:
            from statsmodels.stats.sandwich_covariance import cov_cluster
            clustered_cov = cov_cluster(fit_res, df['dyad'])
            # Create a results object with the clustered covariance matrix used for inference.
            # get_robustcov_results may not exist for this results type; but the results object
            # provides a method to get a new results instance when available. If not, we attach
            # the clustered covariance as an attribute and compute robust bse for printing.
            try:
                clustered_res = fit_res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
            except AttributeError:
                # Attach clustered covariance matrix and compute robust bse
                fit_res.cov_cluster = clustered_cov
                fit_res.bse_cluster = np.sqrt(np.diag(clustered_cov))
                clustered_res = fit_res
        except Exception:
            # If computing clustered covariance fails for any reason, fall back to the plain fit result.
            clustered_res = fit_res

    # Print a brief summary for convenience and return the robust-results object (or plain results if robust not available)
    try:
        # If we have a wrapper that supports summary(), this will work.
        print(clustered_res.summary())
    except Exception:
        # As a fallback, print parameters and (if available) clustered standard errors
        print("Model parameters:")
        print(clustered_res.params)
        if hasattr(clustered_res, 'bse_cluster'):
            print("Clustered standard errors:")
            print(clustered_res.bse_cluster)

    return clustered_res