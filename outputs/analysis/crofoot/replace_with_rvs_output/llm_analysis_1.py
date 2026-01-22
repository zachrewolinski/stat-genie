from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_with_rvs_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist and drop rows with missing critical values
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    df = df.dropna(subset=required)

    # Force numeric types for numeric columns
    num_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=num_cols)

    # Make sure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Relative group size measures
    # Ratio: focal size / other size (continuous, >1 means focal larger)
    df['RelSizeRatio'] = df['n_focal'] / df['n_other']
    # Absolute difference: focal - other (useful for diagnostics)
    df['RelSizeDiff'] = df['n_focal'] - df['n_other']

    # Location advantage: dist_other - dist_focal
    # Positive => contest is relatively closer to the focal group's center (focal advantage)
    df['RelDistance'] = df['dist_other'] - df['dist_focal']

    # Create a coarse categorical location variable for descriptive checks
    # Threshold chosen as 50 meters (reasonable relative to observed distances); tune if needed
    threshold = 50
    df['LocationCategory'] = pd.cut(
        df['RelDistance'],
        bins=[-float('inf'), -threshold, threshold, float('inf')],
        labels=['OtherSide', 'Neutral', 'FocalSide']
    ).astype('category')

    # Relative number of adult males
    df['RelMales'] = df['m_focal'] - df['m_other']

    # Keep dyad as a categorical variable (string) for use in model formula
    df['dyad'] = df['dyad'].astype('category')

    # Keep other descriptive columns that may be useful
    keep_cols = [
        'win', 'RelSizeRatio', 'RelSizeDiff', 'RelDistance', 'LocationCategory', 'RelMales', 'dyad',
        'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'focal', 'other'
    ]
    # Some columns may not exist (f_focal, f_other, focal, other) depending on input; keep those that do
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """Fit a binomial (logistic) regression predicting focal win.

    Primary predictors: RelSizeRatio (relative group size), RelDistance (location advantage).
    Include their interaction to test whether the effect of relative size depends on contest location.
    Control for relative number of males and dyad fixed effects.

    Returns a fitted model object with cluster-robust standard errors clustered by dyad (if possible).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['win', 'RelSizeRatio', 'RelDistance', 'RelMales', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Formula: main effects + interaction + dyad fixed effects
    formula = 'win ~ RelSizeRatio + RelDistance + RelMales + RelSizeRatio:RelDistance + C(dyad)'

    # Fit GLM (logistic)
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Attempt to compute cluster-robust SE clustered on dyad
    try:
        clustered_results = glm_model.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If clustering fails for any reason, return the original model fit
        clustered_results = glm_model

    return clustered_results


