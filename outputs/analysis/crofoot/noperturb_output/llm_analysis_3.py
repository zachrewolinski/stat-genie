from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats as _scipy_stats

# load data (path kept as in original)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/noperturb_output/crofoot.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw data into the final dataframe used for modeling.

    Adds the following columns:
    - RelSize: n_focal - n_other
    - dist_diff: dist_other - dist_focal (positive => contest closer to focal center)
    - DistDiff_z: z-scored version of dist_diff (population std)
    - Location: categorical label (FocalCenter / OtherCenter / Neutral) based on dist_diff threshold
    - MaleDiff: m_focal - m_other
    - RelSize_z, MaleDiff_z: z-scored versions of the above continuous predictors
    - Ensures dyad is categorical and drops rows with missing values in required columns.

    Additionally:
    - Converts 'win' to numeric and drops dyad groups that provide no information
      for estimating dyad fixed effects (i.e., groups with only one outcome or a single observation),
      since such groups cause perfect separation / singularities in logistic regression
      when dyad fixed effects are included.
    """
    # required columns for the analysis
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    # drop rows missing any of the required columns
    df = df.dropna(subset=required).copy()

    # ensure 'win' is numeric (0/1). Coerce non-numeric -> NaN and drop.
    df['win'] = pd.to_numeric(df['win'], errors='coerce')
    df = df.dropna(subset=['win']).copy()
    # If win is boolean, convert to int
    if df['win'].dtype == bool:
        df['win'] = df['win'].astype(int)
    # If it's floating but exactly integer-like, leave as is (logit will coerce), but ensure it's 0/1
    # (do not enforce further here; assume input is correct)

    # compute relative size (focal - other)
    df['RelSize'] = df['n_focal'] - df['n_other']

    # compute distance difference: positive => contest is closer to the focal group's center
    df['dist_diff'] = df['dist_other'] - df['dist_focal']

    # categorical location label
    # threshold chosen to create a 'neutral' zone when distances are similar (here 50 meters); adjust if desired
    thresh = 50.0
    df['Location'] = np.where(df['dist_diff'] > thresh, 'FocalCenter',
                              np.where(df['dist_diff'] < -thresh, 'OtherCenter', 'Neutral'))

    # male difference
    df['MaleDiff'] = df['m_focal'] - df['m_other']

    # z-score continuous predictors (use population std; guard against zero std)
    # Map raw column name -> desired z-column name
    z_map = {
        'RelSize': 'RelSize_z',
        'dist_diff': 'DistDiff_z',  # note: standardized column must be named 'DistDiff_z' per contract
        'MaleDiff': 'MaleDiff_z'
    }
    for raw_col, z_col in z_map.items():
        mean = df[raw_col].mean()
        std = df[raw_col].std(ddof=0)
        if pd.isna(std) or std == 0:
            # if zero variance, center to zero to avoid NaNs; this leaves a constant column (will be handled later)
            df[z_col] = df[raw_col] - mean
        else:
            df[z_col] = (df[raw_col] - mean) / std

    # ensure dyad is categorical (keeps it for formula and clustering)
    df['dyad'] = df['dyad'].astype('category')

    # Drop dyad groups that cannot contribute to estimating dyad fixed effects:
    # - Groups with only a single observation
    # - Groups where 'win' has no variation (all 0 or all 1)
    # These groups can cause perfect separation / singular matrix errors when including C(dyad) as fixed effects.
    group_counts = df.groupby('dyad')['win'].transform('count')
    group_win_variation = df.groupby('dyad')['win'].transform('nunique')
    keep_mask = (group_counts >= 2) & (group_win_variation > 1)
    df = df[keep_mask].copy()

    # After filtering, ensure dyad categorical codes are clean (remove unused categories)
    df['dyad'] = df['dyad'].cat.remove_unused_categories()

    # keep all columns (including the required final columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fits a logistic regression predicting the probability that the focal group won (win = 1).

    Primary model:
      win ~ RelSize_z * DistDiff_z + MaleDiff_z + C(Location) + C(dyad)

    This includes an interaction between relative size and contest-location (continuous) to test whether the effect
    of relative group size depends on where the contest occurred. Dyad is included as a categorical fixed effect,
    and standard errors are clustered by dyad to account for non-independence.

    Returns the model results object with cluster-robust covariance.
    """
    # check that transform has been applied
    required_cols = ['win', 'RelSize_z', 'DistDiff_z', 'MaleDiff_z', 'Location', 'dyad', 'RelSize', 'dist_diff', 'MaleDiff']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # ensure 'win' is numeric
    df = df.copy()
    df['win'] = pd.to_numeric(df['win'], errors='coerce')
    if df['win'].isnull().any():
        raise ValueError("Column 'win' contains non-numeric values after coercion.")

    # specify formula with interaction between relative size and distance-difference (continuous)
    formula = 'win ~ RelSize_z * DistDiff_z + MaleDiff_z + C(Location) + C(dyad)'

    # fit logistic regression; wrap in try/except to provide a clearer error if singularity still occurs
    try:
        logit_res = smf.logit(formula, data=df).fit(disp=False)
    except np.linalg.LinAlgError:
        # If we still hit linear algebra issues, attempt to refit after removing any remaining constant columns
        # (defensive: drop any predictor columns that are constant)
        const_cols = []
        predictors = ['RelSize_z', 'DistDiff_z', 'MaleDiff_z']
        for col in predictors:
            if df[col].std(ddof=0) == 0:
                const_cols.append(col)
        # Build alternative formula without constant predictors (if any)
        alt_parts = ['RelSize_z * DistDiff_z', 'MaleDiff_z', 'C(Location)', 'C(dyad)']
        # remove parts containing constant cols
        if 'RelSize_z' in const_cols or 'DistDiff_z' in const_cols:
            # if either part of interaction is constant, drop the interaction and related main effects appropriately
            alt_parts = [p for p in alt_parts if p not in ['RelSize_z * DistDiff_z']]
            if 'RelSize_z' not in const_cols:
                alt_parts.insert(0, 'RelSize_z')
            if 'DistDiff_z' not in const_cols:
                alt_parts.insert(0, 'DistDiff_z')
        if 'MaleDiff_z' in const_cols:
            alt_parts = [p for p in alt_parts if p != 'MaleDiff_z']
        alt_formula = 'win ~ ' + ' + '.join([p for p in alt_parts if p])
        logit_res = smf.logit(alt_formula, data=df).fit(disp=False)

    # obtain cluster-robust standard errors clustered by dyad
    # Some versions of statsmodels provide get_robustcov_results on result objects, others do not.
    # To be robust across versions, compute clustered covariance explicitly and wrap the results.
    try:
        # preferred when available
        clustered_res = logit_res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
        return clustered_res
    except AttributeError:
        # compute clustered covariance matrix and build a lightweight wrapper
        clustered_cov = cov_cluster(logit_res, df['dyad'])

        class ClusteredResults:
            def __init__(self, base_res, cov_mat):
                self._base = base_res
                self.params = base_res.params.copy()
                # ensure cov_mat is a DataFrame aligned with params
                try:
                    cov_df = pd.DataFrame(cov_mat, index=self.params.index, columns=self.params.index)
                except Exception:
                    cov_df = pd.DataFrame(cov_mat)
                    cov_df.index = self.params.index
                    cov_df.columns = self.params.index
                self._cov = cov_df
                self.cov_params = lambda: self._cov.values
                self.bse = np.sqrt(np.diag(self._cov.values))
                # compute z-values and p-values (normal approximation)
                self.zvalues = self.params.values / self.bse
                self.tvalues = self.zvalues  # name alias
                self.pvalues = 2 * (1 - _scipy_stats.norm.cdf(np.abs(self.zvalues)))
                # make pvalues and bse aligned Series for convenience
                self.bse = pd.Series(self.bse, index=self.params.index)
                self.pvalues = pd.Series(self.pvalues, index=self.params.index)
                self.zvalues = pd.Series(self.zvalues, index=self.params.index)
                self.tvalues = self.zvalues

            def summary(self, *args, **kwargs):
                # fall back to base summary (note: it will show original s.e.; user can inspect self.bse)
                return self._base.summary(*args, **kwargs)

            def as_table(self):
                # convenience: return a DataFrame with params, bse, z, p
                return pd.DataFrame({
                    'coef': self.params,
                    'std_err': self.bse,
                    'z': self.zvalues,
                    'pval': self.pvalues
                })

            def __getattr__(self, item):
                # proxy other attributes to base result
                return getattr(self._base, item)

        return ClusteredResults(logit_res, clustered_cov)


# If this module is imported, do not run model automatically. The df at top is just a convenience for quick interactive use.
# Users should call transform(df) and then model(transformed_df) explicitly.