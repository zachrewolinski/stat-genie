from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats as _stats

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce variables used in the statistical model.

    Produces:
    - size_diff: n_focal - n_other
    - size_ratio: n_focal / n_other (where n_other > 0)
    - loc_diff: dist_other - dist_focal (positive => focal closer to its center relative to other)
    - focal_home_advantage: binary indicator (1 if dist_focal < dist_other, else 0)
    - z-scored versions: size_diff_z, loc_diff_z, male_diff_z, female_diff_z
    - ensures focal and dyad are categorical
    - drops rows with missing values in columns required for the model
    """
    # Required raw columns for the analysis
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'focal', 'dyad'
    ]

    # Work on a copy
    df = df.copy()

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where expected
    numeric_cols = ['n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'win']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    # Relative size measures
    df['size_diff'] = df['n_focal'] - df['n_other']
    # protect division by zero: set ratio to NaN when n_other<=0
    df['size_ratio'] = df.apply(lambda r: (r['n_focal'] / r['n_other']) if r['n_other'] > 0 else np.nan, axis=1)

    # Location measures: positive loc_diff => focal is relatively closer to its center (potential home advantage)
    df['loc_diff'] = df['dist_other'] - df['dist_focal']
    df['focal_home_advantage'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Male / female composition differences
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Z-score continuous predictors for interpretability (use population std ddof=0)
    def zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return series - series.mean()
        return (series - series.mean()) / std

    df['size_diff_z'] = zscore(df['size_diff'])
    df['loc_diff_z'] = zscore(df['loc_diff'])
    df['male_diff_z'] = zscore(df['male_diff'])
    df['female_diff_z'] = zscore(df['female_diff'])

    # Cast focal and dyad to categorical for use as fixed effects / clustering identifiers
    df['focal'] = df['focal'].astype('category')
    df['dyad'] = df['dyad'].astype('category')

    # Re-check that the DV is 0/1 integer
    df['win'] = df['win'].astype(int)

    # Final subset: drop any rows where derived ratios became NaN (e.g., n_other <= 0)
    df = df.dropna(subset=['size_ratio'])

    # Return transformed dataframe with columns used in modeling
    # Keep original columns as well for transparency; key model columns are appended
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting probability that the focal group won the contest.

    Primary model tests:
    - main effects of relative group size (size_diff_z) and contest location (loc_diff_z)
    - interaction size_diff_z * loc_diff_z to test whether location moderates the effect of relative size
    - controls: male_diff_z, female_diff_z, focal fixed effects, dyad used for clustering standard errors

    Returns an object exposing parameter estimates and cluster-robust (by dyad) covariance estimates.
    """
    # Ensure required columns are present
    required = ['win', 'size_diff_z', 'loc_diff_z', 'male_diff_z', 'female_diff_z', 'focal', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula with interaction between relative size and location
    formula = 'win ~ size_diff_z * loc_diff_z + male_diff_z + female_diff_z + C(focal)'

    # Fit logistic regression (maximum likelihood)
    logit_model = smf.logit(formula=formula, data=df)
    results = logit_model.fit(disp=False)

    # Obtain cluster-robust standard errors clustered by dyad (accounts for non-independence within dyad pairs)
    # If dyad is categorical, pass the underlying labels aligned with the data used to fit the model
    cluster_groups = df['dyad'].values

    # Some statsmodels versions provide get_robustcov_results on result objects; use it when available.
    # Otherwise compute clustered covariance matrix and construct a lightweight results wrapper.
    try:
        robust_results = results.get_robustcov_results(cov_type='cluster', groups=cluster_groups)
        return robust_results
    except AttributeError:
        # Compute cluster-robust covariance matrix
        cov = cov_cluster(results, cluster_groups)

        class RobustResultWrapper:
            def __init__(self, base_results, cov_matrix):
                self._base = base_results
                # ensure cov matrix is a numpy array
                cov_arr = np.asarray(cov_matrix)
                # build a DataFrame for cov_params for convenience
                idx = base_results.params.index
                self.cov_params = pd.DataFrame(cov_arr, index=idx, columns=idx)
                self.params = base_results.params
                self.bse = pd.Series(np.sqrt(np.diag(cov_arr)), index=idx)
                # z-statistics and p-values using normal approximation
                z_vals = self.params / self.bse
                self.pvalues = pd.Series(2 * _stats.norm.sf(np.abs(z_vals)), index=idx)
                self.model = base_results.model
                self.df_model = base_results.df_model
                self.df_resid = base_results.df_resid

            def conf_int(self, alpha=0.05):
                q = _stats.norm.ppf(1 - alpha / 2)
                lower = self.params - q * self.bse
                upper = self.params + q * self.bse
                return pd.DataFrame({'lower': lower, 'upper': upper})

            def summary(self):
                # Return the original summary (note: it reflects original SEs). Users can examine params, bse, pvalues here.
                return self._base.summary()

            # expose string representation
            def __repr__(self):
                return f"<RobustResultWrapper params={self.params.to_dict()}>"

        return RobustResultWrapper(results, cov)