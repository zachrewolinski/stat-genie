from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin contest dataset to create the variables used for modeling.

    Produces the following columns (in addition to original):
      - RelGroupSize: n_focal - n_other
      - RelGroupSize_z: standardized RelGroupSize (mean 0, sd 1)
      - ContestLocation: 1 if dist_focal < dist_other (contest closer to focal center), else 0
      - DistDiff: dist_other - dist_focal
      - DistDiff_z: standardized DistDiff (mean 0, sd 1)

    Drops rows with missing values in any variables required for the model.
    """
    # Make a copy to avoid modifying input dataframe in-place
    df = df.copy()

    # Ensure numeric types for the columns used
    numeric_cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'dyad']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create relative group size (focal - other)
    df['RelGroupSize'] = df['n_focal'] - df['n_other']

    # Create contest location indicator: 1 if contest is closer to focal group's center than to other group's center
    # (i.e., dist_focal < dist_other means contest occurred more inside/near focal home range)
    df['ContestLocation'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Distance difference: positive when contest is farther from focal center (closer to other group's center)
    df['DistDiff'] = df['dist_other'] - df['dist_focal']

    # Drop rows missing any of the required columns for modeling
    required = ['win', 'RelGroupSize', 'ContestLocation', 'DistDiff', 'm_focal', 'm_other', 'dyad']
    df = df.dropna(subset=required)

    # Standardize continuous predictors (z-score). Use sample std (ddof=1) to be consistent with common practice.
    # If variance is zero (constant), leave as zero to avoid division by zero.
    for col in ['RelGroupSize', 'DistDiff']:
        mean = df[col].mean()
        std = df[col].std()
        if pd.isna(std) or std == 0:
            df[col + '_z'] = df[col] - mean
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Keep only columns necessary for modeling + identifiers
    keep_cols = ['focal', 'other', 'dyad', 'win', 'dist_focal', 'dist_other',
                 'n_focal', 'n_other', 'm_focal', 'm_other',
                 'RelGroupSize', 'RelGroupSize_z', 'ContestLocation', 'DistDiff', 'DistDiff_z']
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability that the focal group won (win == 1).

    Model specification:
      logit( P(win=1) ) = b0 + b1*RelGroupSize_z + b2*ContestLocation + b3*(RelGroupSize_z * ContestLocation)
                           + b4*DistDiff_z + b5*m_focal + b6*m_other

    We estimate a logit model and compute cluster-robust standard errors clustered by dyad.
    Returns an object containing params, bse, pvalues, cov_params(method=None), and a summary() method.
    """
    # Ensure required transformed columns are present
    required = ['win', 'RelGroupSize_z', 'ContestLocation', 'DistDiff_z', 'm_focal', 'm_other', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Work on a copy to avoid side-effects
    df = df.copy()

    # Build interaction term explicitly (internal helper column)
    df['RelGroupSize_x_Location'] = df['RelGroupSize_z'] * df['ContestLocation']

    # Design matrix
    X_cols = ['RelGroupSize_z', 'ContestLocation', 'RelGroupSize_x_Location', 'DistDiff_z', 'm_focal', 'm_other']
    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['win']

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)

    # Compute cluster-robust covariance (clustered by dyad) if possible,
    # otherwise fall back to HC1 robust covariance.
    try:
        cov = sm.stats.sandwich_covariance.cov_cluster(res, df['dyad'])
    except Exception:
        # Fallback to HC1 covariance
        cov = sm.stats.sandwich_covariance.cov_hc1(res)

    # Construct robust result-like object
    params = res.params.copy()
    param_names = params.index.tolist()
    cov_df = pd.DataFrame(cov, index=param_names, columns=param_names)
    bse = pd.Series(np.sqrt(np.diag(cov_df.values)), index=param_names)
    z_values = params / bse
    pvalues = pd.Series(2 * norm.sf(np.abs(z_values)), index=param_names)

    class RobustResults:
        def __init__(self, params, bse, pvalues, cov_df, z_values):
            self.params = params
            self.bse = bse
            self.pvalues = pvalues
            self._cov = cov_df
            self.tvalues = z_values

        def cov_params(self):
            return self._cov

        def summary(self):
            rows = []
            for name in self.params.index:
                rows.append({
                    'param': name,
                    'coef': float(self.params[name]),
                    'std err': float(self.bse[name]),
                    'z': float(self.tvalues[name]),
                    'P>|z|': float(self.pvalues[name])
                })
            df_sum = pd.DataFrame(rows).set_index('param')
            return df_sum

        def __repr__(self):
            return f"<RobustResults params={self.params.to_dict()}>"

    res_robust = RobustResults(params=params, bse=bse, pvalues=pvalues, cov_df=cov_df, z_values=z_values)

    return res_robust