from typing import Any, Dict
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_and_positive_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin intergroup contest dataframe into the analysis dataframe.

    Produces the following columns (in addition to original ids):
    - rel_size: n_focal - n_other
    - rel_size_z: z-scored rel_size
    - dist_diff: dist_other - dist_focal (positive => contest closer to focal home)
    - dist_diff_z: z-scored dist_diff
    - focal_home: binary 1 if dist_focal < dist_other, else 0
    - male_diff: m_focal - m_other
    - male_diff_z: z-scored male_diff
    - female_diff: f_focal - f_other
    - female_diff_z: z-scored female_diff

    Returns the dataframe with these columns plus win, dyad, focal, other.
    """

    df = df.copy()

    # Drop rows with missing values in columns needed for the analysis
    required_cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    df = df.dropna(subset=required_cols)

    # Relative size (focal - other)
    df['rel_size'] = df['n_focal'] - df['n_other']

    # Distance difference: positive means contest closer to focal's home center
    df['dist_diff'] = df['dist_other'] - df['dist_focal']

    # Binary focal-home indicator (moderator): 1 if contest is closer to focal group's center
    df['focal_home'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Sex composition differences
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Standardize (z-score) continuous predictors used in the model
    for col in ['rel_size', 'dist_diff', 'male_diff', 'female_diff']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            std = 1.0
        df[col + '_z'] = (df[col] - mean) / std

    # Ensure types for id and outcome columns if present
    if 'dyad' in df.columns:
        try:
            df['dyad'] = df['dyad'].astype(int)
        except Exception:
            # If dyad cannot be cast to int (e.g., non-numeric), leave as-is but ensure no missing
            df['dyad'] = df['dyad'].astype(object)
    if 'focal' in df.columns:
        df['focal'] = df['focal'].astype(int)
    if 'other' in df.columns:
        df['other'] = df['other'].astype(int)
    df['win'] = df['win'].astype(int)
    df['focal_home'] = df['focal_home'].astype(int)

    # Keep only columns needed for modeling plus ids for later diagnostics
    keep_cols = ['win', 'rel_size_z', 'dist_diff_z', 'focal_home', 'male_diff_z', 'female_diff_z', 'dyad', 'focal', 'other']
    # If any of the keep columns don't exist (edge case), raise informative error
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required transformed columns: {missing}")

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit a binomial (logistic) regression to test whether relative group size and contest location
    influence the probability that the focal group wins.

    Model formula:
    win ~ rel_size_z * focal_home + dist_diff_z + male_diff_z + female_diff_z

    We fit a GLM with binomial family and calculate dyad-clustered robust standard errors to
    account for non-independence of repeated observations from the same dyad.

    Returns a dictionary containing the fitted model, clustered results (covariance and SEs),
    and a table of odds ratios with 95% confidence intervals.
    """

    # Ensure required columns present
    required = ['win', 'rel_size_z', 'dist_diff_z', 'focal_home', 'male_diff_z', 'female_diff_z', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    formula = 'win ~ rel_size_z * focal_home + dist_diff_z + male_diff_z + female_diff_z'

    # Fit GLM (logistic)
    glm_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute dyad-clustered robust covariance matrix using sandwich estimator
    # Use statsmodels' cov_cluster helper
    try:
        from statsmodels.stats.sandwich_covariance import cov_cluster
        cov = cov_cluster(glm_fit, df['dyad'])
    except Exception:
        # As a fallback compute clustered cov manually using the results' model/exog and groups
        # This fallback attempts to replicate cov_cluster behavior but may be less robust.
        # We'll raise an informative error if cov_cluster is not available or fails.
        raise RuntimeError("Unable to compute clustered covariance matrix using statsmodels.stats.sandwich_covariance.cov_cluster")

    # Standard errors from clustered covariance
    se_cluster = np.sqrt(np.diag(cov))

    # Compute odds ratios and 95% CIs from clustered standard errors
    params = glm_fit.params
    # Use normal approximation for CI (Wald). 97.5 percentile:
    from scipy.stats import norm
    z = norm.ppf(0.975)
    ci_low = params - z * se_cluster
    ci_high = params + z * se_cluster

    or_vals = np.exp(params)
    or_ci_low = np.exp(ci_low)
    or_ci_high = np.exp(ci_high)

    odds_df = pd.DataFrame({
        'OR': or_vals,
        '2.5%': or_ci_low,
        '97.5%': or_ci_high
    })

    # Pack results
    results = {
        'glm_fit': glm_fit,                          # original model fit (for raw summary)
        'glm_clustered': {                           # clustered covariance and SEs
            'cov': cov,
            'se': pd.Series(se_cluster, index=params.index)
        },
        'odds_ratios': odds_df                       # odds ratios with 95% CI (clustered)
    }

    return results