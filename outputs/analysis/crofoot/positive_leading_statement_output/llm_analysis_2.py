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
    Transform the raw capuchin intergroup contest dataframe into a modeling-ready dataframe.

    Produces the following new columns (all names used later in the model code):
      - SizeDiff: n_focal - n_other
      - MaleDiff: m_focal - m_other
      - FemaleDiff: f_focal - f_other
      - DistDiff: dist_other - dist_focal
      - AtHome: binary indicator (1 if dist_focal < dist_other else 0)
      - SizeDiff_z, MaleDiff_z, FemaleDiff_z, DistDiff_z: standardized (z-scored) versions

    Keeps identifying columns focal, other, dyad and the outcome win.
    """
    # Work on a copy
    df = df.copy()

    # Drop rows with missing values in any variables required for transformation/modeling
    required_cols = [
        'win', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other',
        'dist_focal', 'dist_other', 'dyad', 'focal', 'other'
    ]
    df = df.dropna(subset=required_cols)

    # Derived counts/differences
    df['SizeDiff'] = df['n_focal'] - df['n_other']
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Location measures: positive DistDiff means other is farther from its center than focal (i.e., contest relatively closer to focal home)
    df['DistDiff'] = df['dist_other'] - df['dist_focal']

    # Binary at-home indicator: 1 if contest is closer to focal group's home center than to the other's
    df['AtHome'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for stability with small samples.
    for col in ['SizeDiff', 'MaleDiff', 'FemaleDiff', 'DistDiff']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If constant or undefined, create a zero column to avoid division by zero
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Ensure outcome is integer (0/1)
    df['win'] = df['win'].astype(int)

    # Keep only columns needed for modeling (plus ids for potential clustering)
    keep_cols = [
        'win', 'SizeDiff', 'SizeDiff_z', 'AtHome', 'DistDiff', 'DistDiff_z',
        'MaleDiff', 'MaleDiff_z', 'FemaleDiff', 'FemaleDiff_z', 'focal', 'other', 'dyad'
    ]

    # If any keep_cols missing due to earlier issues, add them as NaN so selection doesn't fail
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a generalized estimating equations (GEE) logistic regression to predict the probability
    that the focal group wins an intergroup contest.

    Model formula:
      win ~ SizeDiff_z * AtHome + MaleDiff_z + FemaleDiff_z + DistDiff_z

    Clustering / repeated-measures structure: dyad (pairs of groups) using an exchangeable correlation structure
    to account for non-independence of observations within the same dyad.

    Returns the fitted GEE results object (with summary printed).
    """
    # Local imports (patsy used to build design matrices for sm.GEE)
    import statsmodels.api as sm
    from patsy import dmatrices

    # Drop rows with missing data in the model variables
    model_vars = ['win', 'SizeDiff_z', 'AtHome', 'MaleDiff_z', 'FemaleDiff_z', 'DistDiff_z', 'dyad']
    df_model = df.dropna(subset=model_vars).copy()

    if df_model.shape[0] == 0:
        raise ValueError('No rows available after dropping missing values for model variables.')

    # Build design matrices with patsy (adds intercept automatically)
    formula = 'win ~ SizeDiff_z * AtHome + MaleDiff_z + FemaleDiff_z + DistDiff_z'
    y, X = dmatrices(formula, df_model, return_type='dataframe')

    # Groups for GEE clustering
    groups = df_model['dyad']

    # Specify GEE with binomial family and exchangeable correlation structure
    cov_struct = sm.cov_struct.Exchangeable()
    gee_model = sm.GEE(endog=y, exog=X, groups=groups, family=sm.families.Binomial(), cov_struct=cov_struct)

    # Fit model; use default scale and iteration settings
    result = gee_model.fit()

    # Print and return results for inspection
    print(result.summary())
    return result


