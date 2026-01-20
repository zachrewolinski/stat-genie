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
    Transform the raw dataset into a dataframe containing all columns used in the statistical model.
    New columns created:
      - RelSizeRatio: n_focal / n_other
      - LogRelSizeRatio: log(RelSizeRatio)
      - RelSizeDiff: n_focal - n_other
      - DistDiff: dist_other - dist_focal (positive => focal is relatively closer)
      - FocalCloser: binary 1 if dist_focal < dist_other else 0
      - MaleDiff: m_focal - m_other
      - FemaleDiff: f_focal - f_other
      - *_z standardized (z-score) versions for continuous predictors used in the model
    Rows with missing values in required columns are dropped.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                     'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    df = df.dropna(subset=required_cols)

    # Compute relative size measures
    # n_other should be > 0 based on schema (min 5), but guard against division issues
    df['RelSizeRatio'] = df['n_focal'] / df['n_other']
    # Take log to symmetrize ratio (positive when focal > other, negative when focal < other)
    df['LogRelSizeRatio'] = np.log(df['RelSizeRatio'].replace(0, np.nan))
    df['RelSizeDiff'] = df['n_focal'] - df['n_other']

    # Compute location measures: positive DistDiff means focal is relatively closer
    df['DistDiff'] = df['dist_other'] - df['dist_focal']
    df['FocalCloser'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Sex-composition differences
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Standardize continuous predictors (z-score). Use population std (ddof=0) to avoid small-sample ddof issues.
    to_z = ['LogRelSizeRatio', 'DistDiff', 'MaleDiff', 'FemaleDiff']
    for col in to_z:
        # If column is constant (std==0) avoid divide-by-zero and leave zeros
        std = df[col].std(ddof=0)
        if pd.isna(std) or std == 0:
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - df[col].mean()) / std

    # Keep only columns needed for modeling (plus original useful columns)
    keep_cols = ['win', 'LogRelSizeRatio_z', 'RelSizeDiff', 'FocalCloser',
                 'DistDiff_z', 'MaleDiff_z', 'FemaleDiff_z', 'dyad',
                 # also keep raw counts for possible secondary checks
                 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                 'm_focal', 'm_other', 'f_focal', 'f_other']
    # Some columns (RelSizeDiff) may not be in keep_cols; ensure it's present
    if 'RelSizeDiff' not in keep_cols:
        keep_cols.append('RelSizeDiff')

    # Some datasets might not have exactly the same column ordering; return the full df but ensure model columns exist
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting the probability the focal group wins (win==1).

    Primary test: main effect of relative group size (LogRelSizeRatio_z) and contest location (FocalCloser),
    and their interaction to test whether location moderates the effect of relative group size.

    Controls: DistDiff_z, MaleDiff_z, FemaleDiff_z, and dyad fixed effects (categorical).

    Returns the fitted statsmodels Logit results object.
    """
    import statsmodels.formula.api as smf

    # Ensure we operate on a copy
    df_model = df.copy()

    # Drop any rows with missing model predictors
    model_cols = ['win', 'LogRelSizeRatio_z', 'FocalCloser', 'DistDiff_z', 'MaleDiff_z', 'FemaleDiff_z', 'dyad']
    df_model = df_model.dropna(subset=model_cols)

    # Formula with interaction between relative size and location (FocalCloser acts as moderator)
    formula = 'win ~ LogRelSizeRatio_z * FocalCloser + DistDiff_z + MaleDiff_z + FemaleDiff_z + C(dyad)'

    # Fit logistic regression (binomial). Use maxiter increased if convergence issues arise.
    try:
        res = smf.logit(formula=formula, data=df_model).fit(disp=False, maxiter=200)
    except Exception as e:
        # As a fallback, fit GLM with binomial family (more robust in some cases)
        import statsmodels.api as sm
        res = sm.GLM.from_formula(formula, data=df_model, family=sm.families.Binomial()).fit()

    return res


