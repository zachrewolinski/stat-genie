from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import scipy.stats as stats

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables needed for modeling the probability that the focal group wins.

    Produces the following new columns used in the model:
      - SizeDiff: n_focal - n_other (raw)
      - SizeRatio: n_focal / n_other (raw, kept for diagnostics)
      - LogSizeRatio: log(n_focal / n_other) (raw)
      - RelDist: dist_other - dist_focal (positive -> focal relatively closer to its home center)
      - FocalCloser: boolean indicator (RelDist > 0)
      - MaleDiff: m_focal - m_other (raw)
      - FemaleDiff: f_focal - f_other (raw)
      - z_SizeDiff, z_RelDist, z_MaleDiff, z_FemaleDiff: standardized versions (mean=0, sd=1)

    Rows with missing values for the variables required in the model are dropped.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Derive size-related variables
    df['SizeDiff'] = df['n_focal'] - df['n_other']
    # ratio and log-ratio for diagnostics / alternative specifications
    # avoid division by zero by replacing zeros in denominator with a very small number
    denom = df['n_other'].replace({0: np.finfo(float).eps})
    df['SizeRatio'] = df['n_focal'] / denom
    # guard against non-positive values for log
    df['LogSizeRatio'] = np.log(df['SizeRatio'].replace({0: np.finfo(float).eps}))

    # Derive contest location variables: positive RelDist means focal is closer to its home center
    df['RelDist'] = df['dist_other'] - df['dist_focal']
    df['FocalCloser'] = (df['RelDist'] > 0).astype(int)

    # Sex-composition differences
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Standardize continuous predictors (z-scores). Use sample std (ddof=1) for interpretability.
    def zscore(x: pd.Series) -> pd.Series:
        # If constant, return zeros to avoid NaNs and allow downstream dropping if needed
        if x.std(ddof=1) == 0 or np.isclose(x.std(ddof=1), 0):
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / x.std(ddof=1)

    df['z_SizeDiff'] = zscore(df['SizeDiff'])
    df['z_RelDist'] = zscore(df['RelDist'])
    df['z_MaleDiff'] = zscore(df['MaleDiff'])
    df['z_FemaleDiff'] = zscore(df['FemaleDiff'])

    # Keep only columns needed for modeling (but preserve dyad and win)
    cols_to_keep = [
        'win', 'dyad',
        'SizeDiff', 'SizeRatio', 'LogSizeRatio', 'RelDist', 'FocalCloser',
        'MaleDiff', 'FemaleDiff',
        'z_SizeDiff', 'z_RelDist', 'z_MaleDiff', 'z_FemaleDiff'
    ]

    # Some rows may have become NA if a column had constant values; drop any remaining NAs in kept cols
    df = df[cols_to_keep].dropna()

    # Ensure win is numeric 0/1
    if not pd.api.types.is_numeric_dtype(df['win']):
        df['win'] = pd.to_numeric(df['win'], errors='coerce')
        df = df.dropna(subset=['win'])
    df['win'] = df['win'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability the focal group wins (win==1).

    Primary specification:
      win ~ z_SizeDiff * z_RelDist + z_MaleDiff + z_FemaleDiff

    The interaction term tests whether the effect of relative group size depends on contest location (focal proximity to its home center).

    Returns a statsmodels results object augmented with cluster-robust covariance information clustered by 'dyad'.
    Prints a summary table with clustered standard errors.
    """
    formula = 'win ~ z_SizeDiff * z_RelDist + z_MaleDiff + z_FemaleDiff'

    # Fit logistic regression (maximum likelihood)
    logit_res = smf.logit(formula, data=df).fit(disp=False)

    # Compute cluster-robust covariance matrix clustered by dyad
    # cov_cluster accepts the fitted results and the cluster group labels
    try:
        cluster_cov = cov_cluster(logit_res, df['dyad'])
    except Exception:
        # fallback: try passing numpy array of groups
        cluster_cov = cov_cluster(logit_res, np.asarray(df['dyad']))

    # Clustered standard errors
    clustered_bse = np.sqrt(np.diag(cluster_cov))

    params = logit_res.params
    z_vals = params / clustered_bse
    p_vals = 2 * stats.norm.sf(np.abs(z_vals))
    conf_low = params - 1.96 * clustered_bse
    conf_high = params + 1.96 * clustered_bse

    # Prepare a summary table with clustered SEs
    summary_df = pd.DataFrame({
        'coef': params,
        'cluster_se': clustered_bse,
        'z': z_vals,
        'P>|z|': p_vals,
        '2.5%': conf_low,
        '97.5%': conf_high
    })

    print("Logit model coefficients with cluster-robust standard errors (clustered by 'dyad'):")
    print(summary_df)

    # Attach cluster info to results object for downstream use
    setattr(logit_res, 'cluster_cov', cluster_cov)
    setattr(logit_res, 'clustered_bse', clustered_bse)
    setattr(logit_res, 'clustered_summary', summary_df)

    return logit_res