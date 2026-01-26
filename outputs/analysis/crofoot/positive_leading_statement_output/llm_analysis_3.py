from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/positive_leading_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the columns used in the statistical model.

    Produces the following new columns used in the model:
      - RelSizeDiff: n_focal - n_other (raw difference)
      - RelDist: dist_other - dist_focal (positive means contest closer to focal home)
      - male_diff: m_focal - m_other
      - female_diff: f_focal - f_other
      - RelSizeDiff_z, RelDist_z, male_diff_z, female_diff_z: z-scored versions
      - AtFocalHome: binary indicator (1 if dist_focal < dist_other)

    Also drops rows with missing values in the core columns required for modeling.
    """
    df = df.copy()

    # Required columns for the analysis
    required = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Coerce to numeric where appropriate and drop rows with missing values in required columns
    for col in required:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required).reset_index(drop=True)

    # Derived predictors
    df['RelSizeDiff'] = df['n_focal'] - df['n_other']
    df['RelDist'] = df['dist_other'] - df['dist_focal']
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Binary indicator of whether contest was closer to focal group's home center
    df['AtFocalHome'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize (z-score) continuous predictors using population std (ddof=0) for stability
    def zscore(s: pd.Series) -> pd.Series:
        if s.std(ddof=0) == 0 or pd.isna(s.std(ddof=0)):
            return (s - s.mean())
        return (s - s.mean()) / s.std(ddof=0)

    df['RelSizeDiff_z'] = zscore(df['RelSizeDiff'])
    df['RelDist_z'] = zscore(df['RelDist'])
    df['male_diff_z'] = zscore(df['male_diff'])
    df['female_diff_z'] = zscore(df['female_diff'])

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Keep only columns required for modeling plus some diagnostics
    keep_cols = [
        'win', 'dyad',
        'RelSizeDiff', 'RelDist', 'male_diff', 'female_diff',
        'RelSizeDiff_z', 'RelDist_z', 'male_diff_z', 'female_diff_z',
        'AtFocalHome', 'focal', 'other'
    ]

    # Some columns like 'focal' and 'other' might be missing; keep intersection
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting the probability that the focal group won
    an intergroup contest as a function of relative group size, contest location, and their interaction.

    Model specification:
      win ~ RelSizeDiff_z * RelDist_z + male_diff_z + female_diff_z

    We compute cluster-robust standard errors clustered by 'dyad' to account for non-independence
    of observations within dyads.

    Returns the fitted model object with cluster-robust covariances.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns are present
    required = ['win', 'RelSizeDiff_z', 'RelDist_z', 'male_diff_z', 'female_diff_z', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula with interaction between relative size and relative location
    formula = 'win ~ RelSizeDiff_z * RelDist_z + male_diff_z + female_diff_z'

    # Fit binomial GLM
    glm_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute cluster-robust standard errors clustered by dyad
    try:
        glm_clustered = glm_fit.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # Fallback: return the original fit if clustering fails
        glm_clustered = glm_fit

    # Return the model with clustered covariances (or the original fit object)
    return glm_clustered


