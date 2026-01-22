from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
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
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist and drop rows with missing values in essential columns
    required_cols = ['win','n_focal','n_other','dist_focal','dist_other','m_focal','m_other','f_focal','f_other','dyad']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    numeric_cols = ['win','n_focal','n_other','dist_focal','dist_other','m_focal','m_other','f_focal','f_other','dyad']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    # Primary derived variables
    df['size_diff'] = df['n_focal'] - df['n_other']
    # ratio (add small constant guard if needed)
    df['size_ratio'] = df['n_focal'] / df['n_other'].replace(0, np.nan)

    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Location-related variables
    df['dist_diff'] = df['dist_other'] - df['dist_focal']
    # Binary indicator: focal is closer to its home-range center than other (home advantage)
    df['focal_closer'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize continuous predictors (z-score). Use ddof=0 for population-style std to avoid small-sample issues, but pandas default ddof=1 is also acceptable.
    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std(ddof=0)

    df['size_diff_z'] = zscore(df['size_diff'])
    df['size_ratio_z'] = zscore(df['size_ratio'].fillna(df['size_ratio'].mean()))
    df['m_diff_z'] = zscore(df['m_diff'])
    df['f_diff_z'] = zscore(df['f_diff'])
    df['dist_diff_z'] = zscore(df['dist_diff'])

    # Interaction term between standardized size difference and focal_closer (to test moderation)
    df['size_x_loc'] = df['size_diff_z'] * df['focal_closer']

    # Keep only columns necessary for modeling (but don't drop dyad since we'll cluster on it)
    model_cols = [
        'win',
        'size_diff', 'size_ratio', 'size_diff_z', 'size_ratio_z',
        'm_diff', 'm_diff_z', 'f_diff', 'f_diff_z',
        'dist_diff', 'dist_diff_z', 'focal_closer',
        'size_x_loc', 'dyad', 'focal', 'other'
    ]

    # Some columns (focal, other) may already exist; include them if present
    for c in model_cols:
        if c not in df.columns:
            # If e.g. 'focal' or 'other' not present (should be present per schema), skip
            pass

    # Return dataframe with at least the columns we will use; keep original extras as well
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Prepare design matrix
    # Predictors: standardized size diff, location (binary focal_closer), their interaction,
    # and controls m_diff_z, f_diff_z, dist_diff_z. Cluster SEs by dyad.

    predictors = [
        'size_diff_z',
        'focal_closer',
        'size_x_loc',
        'm_diff_z',
        'f_diff_z',
        'dist_diff_z'
    ]

    # Drop rows with missing values in predictors or outcome
    model_df = df.dropna(subset=['win'] + predictors + ['dyad'])

    X = model_df[predictors]
    X = sm.add_constant(X)
    y = model_df['win']

    # Fit a binomial GLM (logistic regression). Use clustered standard errors by dyad to account for non-independence.
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    results = glm_binom.fit()

    # Obtain clustered robust covariance (cluster by dyad)
    try:
        results_clust = results.get_robustcov_results(cov_type='cluster', groups=model_df['dyad'])
    except Exception:
        # If get_robustcov_results isn't available for the GLMResults object depending on statsmodels version,
        # fallback to refitting with cov_type in fit (some versions allow fit(cov_type='cluster', cov_kwds=...)).
        results = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': model_df['dyad']})
        results_clust = results

    # Return the clustered-results object (contains summary, params, pvalues, conf_int, etc.)
    return results_clust


