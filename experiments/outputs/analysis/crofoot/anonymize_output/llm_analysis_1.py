from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data (kept from original file; can be removed if not desired)
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/anonymize_output/crofoot.csv')
except Exception:
    # If the file is not present in the environment, avoid failing on import.
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # 1) Rename columns to semantically meaningful names (if present)
    df = df.rename(columns={
        'feature1': 'focal_group_id',
        'feature2': 'other_group_id',
        'feature3': 'dyad_id',
        'feature4': 'focal_won',
        'feature5': 'focal_dist_center_m',
        'feature6': 'other_dist_center_m',
        'feature7': 'focal_n_total',
        'feature8': 'other_n_total',
        'feature9': 'focal_n_males',
        'feature10': 'other_n_males',
        'feature11': 'focal_n_females',
        'feature12': 'other_n_females'
    })

    # 2) Drop rows with missing key outcome or size/location predictors
    required_cols = [
        'focal_won',
        'focal_dist_center_m',
        'other_dist_center_m',
        'focal_n_total',
        'other_n_total',
        'focal_n_males',
        'other_n_males',
        'focal_n_females',
        'other_n_females',
        'dyad_id'
    ]
    # Keep only rows that have these required raw inputs (if present in df)
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # If focal_won present, ensure it's numeric
    if 'focal_won' in df.columns:
        df['focal_won'] = pd.to_numeric(df['focal_won'], errors='coerce')

    # 3) Derived predictors for relative group size
    # Ensure raw size columns exist; if not, create them as NaN so later checks will remove rows
    for c in ['focal_n_total', 'other_n_total']:
        if c not in df.columns:
            df[c] = np.nan

    df['relative_group_size'] = df['focal_n_total'] - df['other_n_total']

    # ratio: handle division-by-zero defensively
    denom = df['other_n_total'].replace({0: np.nan})
    df['relative_group_size_ratio'] = df['focal_n_total'] / denom

    # 4) Location-derived variables: distance difference and a binary indicating whether contest is closer to focal group's center
    # We define distance_diff = other_dist_center_m - focal_dist_center_m
    for c in ['focal_dist_center_m', 'other_dist_center_m']:
        if c not in df.columns:
            df[c] = np.nan

    df['distance_diff'] = df['other_dist_center_m'] - df['focal_dist_center_m']
    df['focal_adv_location'] = (df['distance_diff'] > 0).astype(int)

    # 5) Sex composition differences (controls)
    for c in ['focal_n_males', 'other_n_males', 'focal_n_females', 'other_n_females']:
        if c not in df.columns:
            df[c] = np.nan

    df['male_diff'] = df['focal_n_males'] - df['other_n_males']
    df['female_diff'] = df['focal_n_females'] - df['other_n_females']

    # 6) Standardize (z-score) continuous predictors used in the model to aid interpretation and numerical stability
    # Create z-scored columns with the exact required final column names:
    # 'relsize_z' (for relative_group_size),
    # 'distance_diff_z', 'male_diff_z', 'female_diff_z'
    def zscore_series(s: pd.Series) -> pd.Series:
        mean = s.mean()
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            # For constant series: set z=0 for non-missing entries, keep NaN for missing entries
            z = pd.Series(index=s.index, dtype=float)
            non_na = s.notna()
            z.loc[non_na] = 0.0
            z.loc[~non_na] = np.nan
            return z
        else:
            return (s - mean) / std

    df['relsize_z'] = zscore_series(df['relative_group_size'])
    df['distance_diff_z'] = zscore_series(df['distance_diff'])
    df['male_diff_z'] = zscore_series(df['male_diff'])
    df['female_diff_z'] = zscore_series(df['female_diff'])

    # 7) Interaction term: relative group size (z) x focal_adv_location
    # If relsize_z is NA, interaction will be NA; that's intended so dropna can remove incomplete rows.
    df['relsize_x_loc'] = df['relsize_z'] * df['focal_adv_location']

    # 8) Keep only columns necessary for the statistical model and for interpretability
    keep_cols = [
        'focal_won',
        'focal_group_id',
        'other_group_id',
        'dyad_id',
        'focal_n_total',
        'other_n_total',
        'relative_group_size',
        'relative_group_size_ratio',
        'relsize_z',
        'focal_adv_location',
        'distance_diff',
        'distance_diff_z',
        'male_diff',
        'male_diff_z',
        'female_diff',
        'female_diff_z',
        'relsize_x_loc'
    ]

    # Some columns above may not exist if inputs missing; ensure they exist in dataframe (fill with NaN)
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[keep_cols]

    # 9) Drop any rows with NaNs in the core model predictors after transformation
    model_required = [
        'focal_won',
        'relsize_z',
        'focal_adv_location',
        'distance_diff_z',
        'male_diff_z',
        'female_diff_z',
        'dyad_id'
    ]
    df = df.dropna(subset=model_required)

    # Ensure outcome is integer 0/1
    df['focal_won'] = df['focal_won'].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting whether the focal group won.

    Model formula (main): focal_won ~ relsize_z + focal_adv_location + relsize_x_loc + distance_diff_z + male_diff_z + female_diff_z

    relsize_x_loc is the interaction between relative group size (z) and focal_adv_location.
    We cluster standard errors by dyad_id to account for non-independence of observations within dyads.
    """

    # Work on a copy
    data = df.copy()

    # If no rows, return a clear empty-result structure rather than letting statsmodels error on empty arrays
    if data.shape[0] == 0:
        return {
            'results': None,
            'summary_text': 'No data available for modeling after transformation.',
            'data_with_predictions': data
        }

    # Define outcome and predictors
    # Ensure focal_won is present
    if 'focal_won' not in data.columns:
        raise ValueError("Required column 'focal_won' not found in input dataframe.")

    y = data['focal_won']

    predictors = [
        'relsize_z',
        'focal_adv_location',
        'relsize_x_loc',
        'distance_diff_z',
        'male_diff_z',
        'female_diff_z'
    ]

    # Ensure predictors exist; if missing, create as zeros (but this should rarely be needed if transform is used)
    for p in predictors:
        if p not in data.columns:
            data[p] = 0.0

    X = data[predictors]
    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit GLM (Binomial). Use cluster-robust SEs by dyad_id.
    fam = sm.families.Binomial()
    glm_model = sm.GLM(y, X, family=fam)

    # Helper wrapper to present clustered-covariance results with needed methods
    class ClusterResultsWrapper:
        def __init__(self, base_results, cov):
            self._base = base_results
            self.cov = cov
            # params is a pandas Series for convenience
            self.params = getattr(base_results, 'params', None)
            # Compute robust standard errors and related statistics
            self.bse = pd.Series(np.sqrt(np.diag(cov)), index=self.params.index if self.params is not None else None)
            # t-values / z-values
            try:
                self.tvalues = self.params / self.bse
            except Exception:
                self.tvalues = None
            # p-values from normal approximation
            try:
                from scipy import stats
                self.pvalues = 2 * (1 - stats.norm.cdf(np.abs(self.tvalues)))
            except Exception:
                # scipy might not be available; fall back to NaNs
                self.pvalues = pd.Series([np.nan] * len(self.params), index=self.params.index)

        def predict(self, *args, **kwargs):
            return self._base.predict(*args, **kwargs)

        def summary(self):
            # Build a simple textual summary similar to statsmodels' summary tables
            try:
                tbl = pd.DataFrame({
                    'coef': self.params,
                    'std_err': self.bse,
                    'z': self.tvalues,
                    'p': self.pvalues
                })
                header = f"Cluster-robust results (clustered by dyad_id)\nNumber of obs: {int(self._base.nobs) if hasattr(self._base, 'nobs') else data.shape[0]}\n"
                body = tbl.to_string(float_format=lambda x: f"{x:0.4f}")
                text = header + body
            except Exception:
                text = str(self._base)
            # Return an object with as_text() to mimic statsmodels' Summary interface
            class _SummaryText:
                def __init__(self, txt):
                    self._txt = txt

                def as_text(self):
                    return self._txt

            return _SummaryText(text)

        def __str__(self):
            return self.summary().as_text()

    # Function to compute clustered covariance matrix using statsmodels utility
    def compute_cluster_covariance(results_obj, groups):
        try:
            from statsmodels.stats.sandwich_covariance import cov_cluster
            cov = cov_cluster(results_obj, groups)
            return cov
        except Exception:
            # As a very conservative fallback, return the original covariance if available or a diagonal matrix
            try:
                cov_orig = results_obj.cov_params()
                return cov_orig
            except Exception:
                # Diagonal of large variances to avoid zero division
                p = len(results_obj.params)
                return np.diag(np.full(p, 1e6))

    try:
        results = glm_model.fit()
        cov = compute_cluster_covariance(results, data['dyad_id'])
        results_clust = ClusterResultsWrapper(results, cov)
    except Exception:
        # Fallback to Logit if GLM fails to converge
        logit_model = sm.Logit(y, X)
        results = logit_model.fit(disp=False)
        cov = compute_cluster_covariance(results, data['dyad_id'])
        results_clust = ClusterResultsWrapper(results, cov)

    # Summary text
    try:
        summary_text = results_clust.summary().as_text()
    except Exception:
        summary_text = str(results_clust)

    # Compute predicted probabilities where possible
    try:
        data['predicted_prob'] = results_clust.predict(X)
    except Exception:
        data['predicted_prob'] = np.nan

    return {
        'results': results_clust,
        'summary_text': summary_text,
        'data_with_predictions': data
    }