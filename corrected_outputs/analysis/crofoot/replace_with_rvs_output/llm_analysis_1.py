from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin contest dataframe to include derived variables used in modeling.

    Produces the following new columns (all names exact and used in modeling):
      - size_ratio: n_focal / n_other
      - size_diff: n_focal - n_other (kept for inspection, not required by model)
      - dist_advantage: dist_other - dist_focal (positive => focal nearer its home center)
      - AtHome: integer indicator (1 if focal is closer to its home center than other, else 0)
      - m_diff: m_focal - m_other
      - size_ratio_z, dist_adv_z, n_focal_z, m_diff_z: z-scored versions for modeling

    Drops rows with missing values in variables required to compute these.

    Returns a dataframe that contains at minimum the columns required by the model:
      ['win', 'size_ratio_z', 'dist_adv_z', 'n_focal_z', 'm_diff_z', 'dyad']
    It also includes a few helper columns for inspection: size_ratio, size_diff, dist_advantage, m_diff, n_focal.
    """
    df = df.copy()

    # drop rows missing core fields used for derived variables or outcome
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad'
    ]
    df = df.dropna(subset=required_cols)

    # compute relative and absolute size measures
    df['size_ratio'] = df['n_focal'] / df['n_other']
    df['size_diff'] = df['n_focal'] - df['n_other']

    # compute distance advantage: positive means focal is closer to its home center than the other group
    df['dist_advantage'] = df['dist_other'] - df['dist_focal']
    df['AtHome'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # male difference
    df['m_diff'] = df['m_focal'] - df['m_other']

    # z-score continuous predictors (use population std ddof=0 for consistency)
    def zscore(s: pd.Series) -> pd.Series:
        mean = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return s - mean
        return (s - mean) / std

    df['size_ratio_z'] = zscore(df['size_ratio'])
    df['dist_adv_z'] = zscore(df['dist_advantage'])
    df['n_focal_z'] = zscore(df['n_focal'])
    df['m_diff_z'] = zscore(df['m_diff'])

    # Keep required final columns plus some inspection columns
    keep_cols = [
        'win',
        'size_ratio', 'size_ratio_z', 'size_diff',
        'dist_advantage', 'dist_adv_z', 'AtHome',
        'm_diff', 'm_diff_z',
        'n_focal', 'n_focal_z',
        'dyad'
    ]
    existing_keep = [c for c in keep_cols if c in df.columns]
    return df[existing_keep].reset_index(drop=True)


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (GLM with binomial family) predicting the binary contest outcome 'win'.

    Model specification:
      win ~ size_ratio_z + dist_adv_z + size_ratio_z:dist_adv_z + n_focal_z + m_diff_z

    We cluster standard errors by 'dyad' to account for non-independence of observations within dyads.

    Returns the fitted results object with clustered robust covariance.
    """
    # Ensure required predictor columns exist
    required_cols = ['win', 'size_ratio_z', 'dist_adv_z', 'n_focal_z', 'm_diff_z', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare design matrix X and outcome y
    # Compute interaction internally from the required conceptual columns (do not require a separate interaction column)
    X = pd.DataFrame({
        'size_ratio_z': df['size_ratio_z'],
        'dist_adv_z': df['dist_adv_z'],
        'interaction': df['size_ratio_z'] * df['dist_adv_z'],
        'n_focal_z': df['n_focal_z'],
        'm_diff_z': df['m_diff_z'],
    }, index=df.index)

    X = sm.add_constant(X, has_constant='add')
    y = df['win'].astype(float)

    groups = df['dyad'].values

    # Fit GLM binomial with clustered covariance by specifying cov_type and cov_kwds in fit.
    # This avoids using get_robustcov_results which may not be available for GLMResults in some statsmodels versions.
    glm_model = sm.GLM(y, X, family=sm.families.Binomial())
    results = glm_model.fit(cov_type='cluster', cov_kwds={'groups': groups})

    # Print a short summary for users (optional) and return the results object
    try:
        print(results.summary())
    except Exception:
        pass

    return results