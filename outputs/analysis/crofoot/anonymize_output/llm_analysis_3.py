from typing import Any, Dict
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import norm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid mutating input
    df = df.copy()

    # Rename raw feature columns to meaningful names (assumes incoming raw columns are named feature1..feature12)
    # If the input already uses the target column names, this will simply convert types below.
    if 'feature1' in df.columns:
        df['focal_group_id'] = df['feature1'].astype(int)
    elif 'focal_group_id' in df.columns:
        df['focal_group_id'] = df['focal_group_id'].astype(int)
    else:
        raise KeyError("Input dataframe must contain 'feature1' or 'focal_group_id'")

    if 'feature2' in df.columns:
        df['other_group_id'] = df['feature2'].astype(int)
    elif 'other_group_id' in df.columns:
        df['other_group_id'] = df['other_group_id'].astype(int)
    else:
        raise KeyError("Input dataframe must contain 'feature2' or 'other_group_id'")

    if 'feature3' in df.columns:
        df['dyad_id'] = df['feature3'].astype(int)
    elif 'dyad_id' in df.columns:
        df['dyad_id'] = df['dyad_id'].astype(int)
    else:
        raise KeyError("Input dataframe must contain 'feature3' or 'dyad_id'")

    if 'feature4' in df.columns:
        df['focal_won'] = df['feature4'].astype(int)
    elif 'focal_won' in df.columns:
        df['focal_won'] = df['focal_won'].astype(int)
    else:
        raise KeyError("Input dataframe must contain 'feature4' or 'focal_won'")

    # Distances from each group's home range center
    if 'feature5' in df.columns:
        df['focal_dist_from_home'] = pd.to_numeric(df['feature5'], errors='coerce')
    else:
        df['focal_dist_from_home'] = pd.to_numeric(df.get('focal_dist_from_home'), errors='coerce')

    if 'feature6' in df.columns:
        df['other_dist_from_home'] = pd.to_numeric(df['feature6'], errors='coerce')
    else:
        df['other_dist_from_home'] = pd.to_numeric(df.get('other_dist_from_home'), errors='coerce')

    # Group sizes (total and by sex)
    if 'feature7' in df.columns:
        df['focal_n_total'] = pd.to_numeric(df['feature7'], errors='coerce').astype('Int64')
    else:
        df['focal_n_total'] = pd.to_numeric(df.get('focal_n_total'), errors='coerce').astype('Int64')

    if 'feature8' in df.columns:
        df['other_n_total'] = pd.to_numeric(df['feature8'], errors='coerce').astype('Int64')
    else:
        df['other_n_total'] = pd.to_numeric(df.get('other_n_total'), errors='coerce').astype('Int64')

    if 'feature9' in df.columns:
        df['focal_n_males'] = pd.to_numeric(df['feature9'], errors='coerce').astype('Int64')
    else:
        df['focal_n_males'] = pd.to_numeric(df.get('focal_n_males'), errors='coerce').astype('Int64')

    if 'feature10' in df.columns:
        df['other_n_males'] = pd.to_numeric(df['feature10'], errors='coerce').astype('Int64')
    else:
        df['other_n_males'] = pd.to_numeric(df.get('other_n_males'), errors='coerce').astype('Int64')

    if 'feature11' in df.columns:
        df['focal_n_females'] = pd.to_numeric(df['feature11'], errors='coerce').astype('Int64')
    else:
        df['focal_n_females'] = pd.to_numeric(df.get('focal_n_females'), errors='coerce').astype('Int64')

    if 'feature12' in df.columns:
        df['other_n_females'] = pd.to_numeric(df['feature12'], errors='coerce').astype('Int64')
    else:
        df['other_n_females'] = pd.to_numeric(df.get('other_n_females'), errors='coerce').astype('Int64')

    # Drop rows with missing values in key variables
    required_cols = [
        'focal_group_id', 'other_group_id', 'dyad_id', 'focal_won',
        'focal_dist_from_home', 'other_dist_from_home',
        'focal_n_total', 'other_n_total',
        'focal_n_males', 'other_n_males',
        'focal_n_females', 'other_n_females'
    ]
    df = df.dropna(subset=required_cols)

    # Compute relative size (focal - other) and standardize
    df['rel_size'] = df['focal_n_total'].astype(float) - df['other_n_total'].astype(float)
    rel_size_std = df['rel_size'].std(ddof=0)
    if rel_size_std == 0 or np.isnan(rel_size_std):
        rel_size_std = 1.0
    df['rel_size_z'] = (df['rel_size'] - df['rel_size'].mean()) / rel_size_std

    # Compute distance difference: other_dist - focal_dist. Positive => focal closer to its home than other (location advantage)
    df['dist_diff'] = df['other_dist_from_home'].astype(float) - df['focal_dist_from_home'].astype(float)
    dist_diff_std = df['dist_diff'].std(ddof=0)
    if dist_diff_std == 0 or np.isnan(dist_diff_std):
        dist_diff_std = 1.0
    df['dist_diff_z'] = (df['dist_diff'] - df['dist_diff'].mean()) / dist_diff_std

    # Sex composition differences, standardized
    df['males_diff'] = df['focal_n_males'].astype(float) - df['other_n_males'].astype(float)
    males_diff_std = df['males_diff'].std(ddof=0)
    if males_diff_std == 0 or np.isnan(males_diff_std):
        males_diff_std = 1.0
    df['males_diff_z'] = (df['males_diff'] - df['males_diff'].mean()) / males_diff_std

    df['females_diff'] = df['focal_n_females'].astype(float) - df['other_n_females'].astype(float)
    females_diff_std = df['females_diff'].std(ddof=0)
    if females_diff_std == 0 or np.isnan(females_diff_std):
        females_diff_std = 1.0
    df['females_diff_z'] = (df['females_diff'] - df['females_diff'].mean()) / females_diff_std

    # Optional categorical contest location label (not required for primary model but useful for exploration)
    # If focal is meaningfully closer (dist_diff > 50) -> 'FocalHome'; if other closer -> 'OtherHome'; else 'Neutral'
    df['contest_location'] = df['dist_diff'].apply(lambda x: 'FocalHome' if x > 50 else ('OtherHome' if x < -50 else 'Neutral'))

    # Keep only the columns needed for modeling and interpretation
    keep_cols = [
        'focal_group_id', 'other_group_id', 'dyad_id', 'focal_won',
        'focal_dist_from_home', 'other_dist_from_home',
        'focal_n_total', 'other_n_total',
        'focal_n_males', 'other_n_males', 'focal_n_females', 'other_n_females',
        'rel_size', 'rel_size_z', 'dist_diff', 'dist_diff_z',
        'males_diff', 'males_diff_z', 'females_diff', 'females_diff_z',
        'contest_location'
    ]

    # Ensure all keep_cols exist (some like raw features may not if input already contained final names)
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[keep_cols].reset_index(drop=True)
    return df


def model(df: pd.DataFrame) -> Any:
    # Formula: main effects of relative size and location (distance difference) plus their interaction.
    # Controls: sex-composition differences and group fixed effects for focal and other groups.
    formula = ('focal_won ~ rel_size_z * dist_diff_z '
               '+ males_diff_z + females_diff_z '
               '+ C(focal_group_id) + C(other_group_id)')

    # Fit logistic regression (binomial logit)
    logit_model = smf.logit(formula=formula, data=df)
    fit_res = logit_model.fit(disp=False)

    # Obtain cluster-robust standard errors clustered by dyad_id
    # Some statsmodels results classes do not implement get_robustcov_results.
    # We compute clustered covariance manually and wrap minimal result info.
    try:
        # Attempt to compute cluster-robust covariance using statsmodels helper
        from statsmodels.stats.sandwich_covariance import cov_cluster
        cov = cov_cluster(fit_res, df['dyad_id'])
    except Exception:
        # Fallback to using the model's covariance (non-robust) if cluster computation fails
        cov = fit_res.cov_params()

    # Build a minimal clustered-results wrapper that exposes params, bse, pvalues, and a simple summary()
    class ClusteredResults:
        def __init__(self, params: pd.Series, cov_matrix: np.ndarray):
            self.params = params
            self._cov = cov_matrix
            # Ensure covariance shape matches params
            try:
                self.bse = np.sqrt(np.diag(self._cov))
            except Exception:
                # fallback: use original bse
                self.bse = fit_res.bse.values if hasattr(fit_res, 'bse') else np.sqrt(np.abs(params.values))
            # z-stats and p-values (two-sided)
            self.z = self.params.values / self.bse
            self.pvalues = 2.0 * (1.0 - norm.cdf(np.abs(self.z)))

        def cov_params(self) -> np.ndarray:
            return self._cov

        def summary(self) -> pd.DataFrame:
            df_tbl = pd.DataFrame({
                'coef': self.params,
                'std err': self.bse,
                'z': self.z,
                'P>|z|': self.pvalues
            }, index=self.params.index)
            return df_tbl

        def __repr__(self) -> str:
            return self.summary().to_string()

    clustered_res = ClusteredResults(fit_res.params, cov)

    # Print brief summaries (user can inspect full results returned)
    print('Logit coefficients (default SE):')
    print(fit_res.summary())
    print('Cluster-robust results (clustered by dyad_id if available):')
    print(clustered_res.summary())

    # Return both the fitted model and clustered results for downstream use
    return {
        'fit_result': fit_res,
        'clustered_result': clustered_res
    }