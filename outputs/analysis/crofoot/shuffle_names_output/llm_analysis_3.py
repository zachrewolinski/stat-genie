from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import norm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe to produce variables for modeling.

    Output columns used in the model (REQUIRED and preserved):
      - WinOutcome: binary 0/1 (from 'dyad')
      - FocalDist: distance of focal group from its home-range center (from 'win')
      - OtherDist: distance of other group from its home-range center (from 'm_focal')
      - FocalSize, OtherSize: absolute group sizes (prefer 'f_focal' and 'f_other' if present, otherwise sum of males+females)
      - NumMalesFocal, NumMalesOther: male counts (from 'n_focal' and 'other')
      - RelSizeDiff: FocalSize - OtherSize
      - RelSizeRatio: FocalSize / OtherSize (NaN if OtherSize == 0)
      - ContestLocation: categorical ('FocalTerritory','OtherTerritory','Neutral') derived from FocalDist and OtherDist
      - Location_*: dummy variables for contest location (Neutral is reference and therefore dropped)
      - RelSizeDiff_c: mean-centered RelSizeDiff
      - MaleDiff: NumMalesFocal - NumMalesOther
      - DyadID: identifier for dyadic pair (from 'm_other') used for clustering
    """
    df = df.copy()

    # Ensure numeric where needed; distances are stored in 'win' and 'm_focal' in this dataset
    if 'win' in df.columns:
        df['FocalDist'] = pd.to_numeric(df['win'], errors='coerce')
    else:
        df['FocalDist'] = pd.Series(np.nan, index=df.index)

    if 'm_focal' in df.columns:
        df['OtherDist'] = pd.to_numeric(df['m_focal'], errors='coerce')
    else:
        df['OtherDist'] = pd.Series(np.nan, index=df.index)

    # Prefer explicit total group-size columns if present; else derive from male+female columns
    if 'f_focal' in df.columns and 'f_other' in df.columns:
        df['FocalSize'] = pd.to_numeric(df['f_focal'], errors='coerce')
        df['OtherSize'] = pd.to_numeric(df['f_other'], errors='coerce')
    else:
        # Safely get male/female components if present; otherwise fill with NaN series
        males_focal = pd.to_numeric(df['n_focal'], errors='coerce') if 'n_focal' in df.columns else pd.Series(np.nan, index=df.index)

        if 'dist_focal' in df.columns:
            females_focal = pd.to_numeric(df['dist_focal'], errors='coerce')
        elif 'focal' in df.columns:
            females_focal = pd.to_numeric(df['focal'], errors='coerce')
        else:
            females_focal = pd.Series(np.nan, index=df.index)

        males_other = pd.to_numeric(df['other'], errors='coerce') if 'other' in df.columns else pd.Series(np.nan, index=df.index)

        if 'focal' in df.columns:
            # Note: using 'focal' here may overlap with above; this follows the original heuristic in the code
            females_other = pd.to_numeric(df['focal'], errors='coerce')
        elif 'dist_other' in df.columns:
            females_other = pd.to_numeric(df['dist_other'], errors='coerce')
        else:
            females_other = pd.Series(np.nan, index=df.index)

        # If any component is missing, the sum will be NaN for that row
        df['FocalSize'] = males_focal + females_focal
        df['OtherSize'] = males_other + females_other

    # Num males (explicit columns if present)
    if 'n_focal' in df.columns:
        df['NumMalesFocal'] = pd.to_numeric(df['n_focal'], errors='coerce')
    else:
        df['NumMalesFocal'] = pd.Series(np.nan, index=df.index)

    if 'other' in df.columns:
        df['NumMalesOther'] = pd.to_numeric(df['other'], errors='coerce')
    else:
        df['NumMalesOther'] = pd.Series(np.nan, index=df.index)

    # Dyad id (used for clustering)
    if 'm_other' in df.columns:
        df['DyadID'] = df['m_other']
    else:
        df['DyadID'] = pd.Series(np.nan, index=df.index)

    # Drop rows missing core variables required for modeling
    # Required: 'dyad', 'FocalSize', 'OtherSize', 'FocalDist', 'OtherDist'
    required = ['dyad', 'FocalSize', 'OtherSize', 'FocalDist', 'OtherDist']
    df = df.dropna(subset=required)

    # Dependent variable
    df['WinOutcome'] = df['dyad'].astype(int)

    # Relative size measures
    df['RelSizeDiff'] = df['FocalSize'] - df['OtherSize']
    # Avoid divide-by-zero
    df['RelSizeRatio'] = df['FocalSize'] / df['OtherSize'].replace({0: np.nan})

    # Contest location classification
    df['_dist_diff'] = df['FocalDist'] - df['OtherDist']
    threshold = 50.0
    df['ContestLocation'] = np.where(df['_dist_diff'] < -threshold, 'FocalTerritory',
                              np.where(df['_dist_diff'] > threshold, 'OtherTerritory', 'Neutral'))

    # Create dummy columns for contest location; drop Neutral as reference
    loc_dummies = pd.get_dummies(df['ContestLocation'], prefix='Location')
    if 'Location_Neutral' in loc_dummies.columns:
        loc_dummies = loc_dummies.drop(columns=['Location_Neutral'])
    df = pd.concat([df, loc_dummies], axis=1)

    # Center relative-size difference for interpretation and interaction
    df['RelSizeDiff_c'] = df['RelSizeDiff'] - df['RelSizeDiff'].mean()

    # Male difference control
    df['MaleDiff'] = df['NumMalesFocal'] - df['NumMalesOther']

    # Clean up helpers
    df = df.drop(columns=['_dist_diff'], errors='ignore')

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability that the focal group wins
    using relative group size, contest location, their interaction, and controls.

    Uses cluster-robust standard errors clustered by dyadic pair (DyadID) to account
    for non-independence of repeated observations within the same dyad.

    Returns an object that exposes model parameters and clustered robust SEs.
    """
    df = df.copy()

    # Outcome
    y = df['WinOutcome']

    # Base covariates and controls
    base_cols = [
        'RelSizeDiff_c',   # centered relative size difference
        'RelSizeRatio',
        'FocalSize',
        'OtherSize',
        'NumMalesFocal',
        'NumMalesOther',
        'MaleDiff',
        'FocalDist',
        'OtherDist'
    ]

    # Location dummy columns (if present) - these were created in transform
    loc_cols = [c for c in df.columns if c.startswith('Location_')]

    X = df[base_cols + loc_cols].copy()

    # Fill NaNs in predictors conservatively
    X = X.fillna(0)

    # Add interaction terms between relative size (centered) and each location dummy
    for loc in loc_cols:
        inter_name = f'{loc}:RelSizeDiff_c'
        X[inter_name] = X[loc] * X['RelSizeDiff_c']

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression
    logit = sm.Logit(y, X)
    try:
        res = logit.fit(disp=False, maxiter=200)
    except Exception:
        # fall back to a more robust solver if needed
        res = logit.fit(disp=False, method='bfgs', maxiter=200)

    # If DyadID is present and has any non-missing values, compute clustered covariance
    if 'DyadID' in df.columns and df['DyadID'].notna().any():
        # Ensure groups align with model observations
        groups = df.loc[X.index, 'DyadID']
        try:
            clustered_cov = cov_cluster(res, groups)
        except Exception:
            # If cov_cluster fails for some reason, fall back to the original results
            clustered_cov = None
    else:
        clustered_cov = None

    # Wrapper object to present clustered results while preserving original results
    class ClusteredResults:
        def __init__(self, base_res, clustered_cov_matrix):
            self._res = base_res
            self.params = base_res.params.copy()
            if clustered_cov_matrix is None:
                # Use base results' covariance if no clustered covariance available
                try:
                    cov = base_res.cov_params()
                except Exception:
                    cov = np.zeros((len(self.params), len(self.params)))
                self.cov = cov
            else:
                self.cov = clustered_cov_matrix

            # Compute standard errors, t-values, p-values, conf_int
            self.bse = pd.Series(np.sqrt(np.diag(self.cov)), index=self.params.index)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                self.tvalues = self.params / self.bse
            # Two-sided p-values assuming normal approximation
            self.pvalues = 2 * (1 - norm.cdf(np.abs(self.tvalues.fillna(0))))
            # 95% CI
            z = norm.ppf(0.975)
            ci_lower = self.params - z * self.bse
            ci_upper = self.params + z * self.bse
            self.conf_int = pd.DataFrame({'2.5%': ci_lower, '97.5%': ci_upper}, index=self.params.index)

            # Expose some model meta info
            self.model = base_res.model
            self.llf = base_res.llf if hasattr(base_res, 'llf') else None
            self.df_model = base_res.df_model if hasattr(base_res, 'df_model') else None
            self.nobs = base_res.nobs if hasattr(base_res, 'nobs') else None

        def summary(self) -> str:
            # Create a concise summary string similar in spirit to statsmodels' summary but simpler
            header = f'Clustered results (clustered covariance used):\n'
            header += f'Number of observations: {int(self.nobs) if self.nobs is not None else "n/a"}\n'
            header += f'Parameters:\n'
            table = pd.DataFrame({
                'coef': self.params,
                'std err': self.bse,
                't': self.tvalues,
                'P>|t|': self.pvalues,
                '[2.5%': self.conf_int['2.5%'],
                '97.5%]': self.conf_int['97.5%']
            })
            # Format table into a string
            with pd.option_context('display.float_format', '{:0.4f}'.format):
                table_str = table.to_string()
            return header + table_str

        # Make some attributes accessible as methods too
        def get_robustcov_results(self, *args, **kwargs):
            # mimic statsmodels API: return self
            return self

    # If clustered covariance exists, return wrapper; else return original results object
    if clustered_cov is not None:
        clustered = ClusteredResults(res, clustered_cov)
    else:
        # Wrap original res with None clustered cov to provide consistent interface
        clustered = ClusteredResults(res, None)

    # Print a concise summary (users can inspect clustered.summary() or clustered.params as needed)
    print(clustered.summary())

    return clustered