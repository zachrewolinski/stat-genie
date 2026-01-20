from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw contest dataset into analysis-ready dataframe.

    Produces the following new columns used in modeling:
      - RelGroupSize: n_focal - n_other (integer)
      - RelGroupSize_z: standardized RelGroupSize (mean 0, sd 1)
      - RelDistance: dist_other - dist_focal (positive => contest closer to focal home center)
      - RelDistance_z: standardized RelDistance
      - MaleDiff: m_focal - m_other
      - MaleDiff_z: standardized MaleDiff
      - FemaleDiff: f_focal - f_other
      - FemaleDiff_z: standardized FemaleDiff
      - TotalSize: n_focal + n_other
      - TotalSize_z: standardized TotalSize
      - FocalHome: binary indicator (1 if dist_focal < dist_other else 0)

    Drops rows with missing values in required columns.
    """

    # Work on a copy
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'win', 'dist_focal', 'dist_other',
        'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Drop rows with missing required values
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where appropriate
    numeric_cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Re-drop if numeric conversion introduced NaNs
    df = df.dropna(subset=numeric_cols + ['dyad'])

    # Cast dyad to a categorical-friendly type (keep numeric but may be treated as categorical in formula)
    df['dyad'] = df['dyad'].astype(int)

    # Compute relative group size (focal - other) and related summaries
    df['RelGroupSize'] = df['n_focal'] - df['n_other']
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Compute difference in distances: positive => contest closer to focal home center
    df['RelDistance'] = df['dist_other'] - df['dist_focal']

    # Male and female differences
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Binary indicator whether contest is closer to focal home center
    df['FocalHome'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize continuous predictors (z-score). Use sample mean/std (ddof=0) for interpretability.
    def zscore(col):
        mu = np.nanmean(col)
        sigma = np.nanstd(col)
        if sigma == 0 or np.isnan(sigma):
            return (col - mu)  # constant column -> zero-centered
        return (col - mu) / sigma

    df['RelGroupSize_z'] = zscore(df['RelGroupSize'].values)
    df['RelDistance_z'] = zscore(df['RelDistance'].values)
    df['MaleDiff_z'] = zscore(df['MaleDiff'].values)
    df['FemaleDiff_z'] = zscore(df['FemaleDiff'].values)
    df['TotalSize_z'] = zscore(df['TotalSize'].values)

    # Keep only columns needed for modeling and useful identifiers
    keep_cols = [
        'win',
        'RelGroupSize', 'RelGroupSize_z',
        'RelDistance', 'RelDistance_z',
        'MaleDiff', 'MaleDiff_z',
        'FemaleDiff', 'FemaleDiff_z',
        'TotalSize', 'TotalSize_z',
        'FocalHome',
        'dyad', 'focal', 'other'
    ]

    # There may be extra columns in df; return the reduced dataframe
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) regression predicting the probability that the focal group wins.

    Model specification (primary):
      win ~ RelGroupSize_z * RelDistance_z + MaleDiff_z + FemaleDiff_z + TotalSize_z + C(dyad)

    - The interaction term tests whether the effect of relative group size depends on contest location (home-range advantage).
    - Dyad is included as a categorical fixed effect to control for pair-specific heterogeneity.

    Returns the fitted GLM results object.
    """
    import statsmodels.formula.api as smf

    # Ensure necessary columns exist
    required = ['win', 'RelGroupSize_z', 'RelDistance_z', 'MaleDiff_z', 'FemaleDiff_z', 'TotalSize_z', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula with interaction between relative group size and relative distance
    formula = 'win ~ RelGroupSize_z * RelDistance_z + MaleDiff_z + FemaleDiff_z + TotalSize_z + C(dyad)'

    # Fit GLM with binomial family (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return fitted model object (results)
    return model


