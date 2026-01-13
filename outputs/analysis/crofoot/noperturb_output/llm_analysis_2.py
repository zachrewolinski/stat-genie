from typing import Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin contest dataframe into the analysis-ready dataframe.

    Produces the following new columns used in the model:
      - SizeAdv_z: standardized (z) of (n_focal - n_other)
      - DistDiff_z: standardized (z) of (dist_other - dist_focal)
      - MaleAdv_z: standardized (z) of (m_focal - m_other)
      - TotalSize_z: standardized (z) of (n_focal + n_other)
      - LocationCat: categorical label of contest location: 'FocalHome', 'OtherHome', or 'Neutral'

    Keeps original 'win' and 'dyad' columns.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure numeric types
    numeric_cols = ['dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in key columns
    df = df.dropna(subset=['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'dyad'])

    # Basic derived variables
    df['SizeAdv'] = df['n_focal'] - df['n_other']
    # A positive DistDiff means dist_other > dist_focal -> contest is closer to focal group's home center
    df['DistDiff'] = df['dist_other'] - df['dist_focal']
    df['MaleAdv'] = df['m_focal'] - df['m_other']
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Create a simple categorical location label to capture nonlinearity / thresholds
    # If the distances are very similar (within 20 meters) we label as 'Neutral'
    # Otherwise, DistDiff > 20 -> 'FocalHome' (contest closer to focal center)
    # DistDiff < -20 -> 'OtherHome' (contest closer to other group's center)
    def categorize_location(x, tol=20.0):
        if pd.isna(x):
            return pd.NA
        if abs(x) <= tol:
            return 'Neutral'
        return 'FocalHome' if x > 0 else 'OtherHome'

    df['LocationCat'] = df['DistDiff'].apply(categorize_location)

    # Standardize (z-score) the continuous predictors used in models
    for raw_col, z_col in [('SizeAdv', 'SizeAdv_z'), ('DistDiff', 'DistDiff_z'), ('MaleAdv', 'MaleAdv_z'), ('TotalSize', 'TotalSize_z')]:
        col = df[raw_col]
        # Use population std (ddof=0) to be explicit; fall back safely when constant
        std = col.std(ddof=0)
        mean = col.mean()
        if std == 0 or pd.isna(std):
            df[z_col] = 0.0
        else:
            df[z_col] = (col - mean) / std

    # Ensure win is integer (0/1)
    df['win'] = df['win'].astype(int)

    # Ensure dyad is integer/categorical for clustering
    df['dyad'] = df['dyad'].astype(int)

    # Drop any rows that may have become NA in categorical LocationCat
    df = df.dropna(subset=['LocationCat'])

    # Final columns required for modeling: keep them and return
    required_cols = ['win', 'SizeAdv_z', 'DistDiff_z', 'MaleAdv_z', 'TotalSize_z', 'LocationCat', 'dyad']
    # If any required column missing for a row, drop that row
    df = df.dropna(subset=required_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability that the focal group wins.

    Model specification:
      win ~ SizeAdv_z * DistDiff_z + MaleAdv_z + TotalSize_z + C(LocationCat)

    We also compute cluster-robust standard errors clustered on dyad to account for repeated contests
    between the same pair of groups.

    Returns a dictionary with the fitted results and the cluster-robust results.
    """
    # Ensure required columns exist
    needed = ['win', 'SizeAdv_z', 'DistDiff_z', 'MaleAdv_z', 'TotalSize_z', 'LocationCat', 'dyad']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit a logistic regression (use discrete Logit as an equivalent to GLM(binomial))
    formula = 'win ~ SizeAdv_z * DistDiff_z + MaleAdv_z + TotalSize_z + C(LocationCat)'
    logit_res = smf.logit(formula=formula, data=df).fit(disp=False)

    # Compute cluster-robust covariance matrix clustered by dyad
    clustered_cov = cov_cluster(logit_res, df['dyad'])

    # Derive clustered standard errors, z-stats, and p-values (normal approximation)
    params = logit_res.params
    clustered_se = np.sqrt(np.diag(clustered_cov))
    with np.errstate(divide='ignore', invalid='ignore'):
        z_cluster = params / clustered_se
    p_cluster = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_cluster)))

    clustered_summary = pd.DataFrame({
        'coef': params,
        'se_cluster': clustered_se,
        'z_cluster': z_cluster,
        'p_cluster': p_cluster
    })

    # Return both objects so the analyst can inspect coefficients and robust summaries
    return {
        'glm_results': logit_res,
        'clustered_cov': clustered_cov,
        'clustered_summary': clustered_summary
    }