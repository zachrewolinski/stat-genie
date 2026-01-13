from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Produces the following new columns used by the model:
      - Win: binary outcome (int) matching original 'win'
      - LogSizeRatio_z: standardized log(n_focal / n_other)
      - HomeAdvantage_z: standardized (dist_other - dist_focal)
      - MaleAdvantage_z: standardized (m_focal - m_other)
      - TotalSize_z: standardized (n_focal + n_other)
      - dyad: kept as provided (used as categorical fixed effect)
    """
    # operate on a copy
    df = df.copy()

    # Keep only rows with the essential columns present
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    for col in ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']:
        # coerce to numeric when possible
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required_cols)  # drop again if coercion produced NaN

    # Outcome (make sure it's integer 0/1)
    df['Win'] = df['win'].astype(int)

    # Compute relative size as log ratio. Avoid division by zero by filtering (n_other>0)
    df = df[df['n_other'] > 0]
    df['LogSizeRatio'] = np.log(df['n_focal'] / df['n_other'])

    # Compute home advantage: positive when focal is closer to its home center than the other group
    df['HomeAdvantage'] = df['dist_other'] - df['dist_focal']

    # Male advantage
    df['MaleAdvantage'] = df['m_focal'] - df['m_other']

    # Total contest/group size
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Standardize (z-score) continuous predictors used in the model
    def zscore(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return series - mean  # will be all zeros
        return (series - mean) / std

    df['LogSizeRatio_z'] = zscore(df['LogSizeRatio'])
    df['HomeAdvantage_z'] = zscore(df['HomeAdvantage'])
    df['MaleAdvantage_z'] = zscore(df['MaleAdvantage'])
    df['TotalSize_z'] = zscore(df['TotalSize'])

    # Ensure dyad is present (kept as numeric or categorical); modeling will treat it as categorical
    df['dyad'] = df['dyad'].astype(int)

    # Return only the columns necessary for modeling plus a few originals for traceability
    keep_cols = ['Win', 'LogSizeRatio_z', 'HomeAdvantage_z', 'MaleAdvantage_z', 'TotalSize_z', 'dyad',
                 'n_focal', 'n_other', 'm_focal', 'm_other', 'dist_focal', 'dist_other']
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) regression to estimate how relative group size and contest location
    influence the probability that the focal group wins.

    The model estimated is a logistic GLM with dyad fixed effects and an interaction between
    relative size and home advantage:
        Win ~ LogSizeRatio_z * HomeAdvantage_z + MaleAdvantage_z + TotalSize_z + C(dyad)

    We then compute cluster-robust standard errors (clustered by dyad) to account for within-dyad
    correlation across contests.

    Returns:
      - results_robust: statsmodels results object with cluster-robust covariance
    """
    import statsmodels.formula.api as smf

    df = df.copy()

    # Formula with interaction between relative size and home advantage
    formula = 'Win ~ LogSizeRatio_z * HomeAdvantage_z + MaleAdvantage_z + TotalSize_z + C(dyad)'

    # Fit GLM (binomial / logistic)
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Obtain cluster-robust covariance by dyad
    # Use the original dyad column as the clustering variable
    try:
        results_robust = glm_model.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # fallback to the original (non-robust) model if robust covariance fails
        results_robust = glm_model

    # Print brief summary and return results object
    print(results_robust.summary())
    return results_robust


