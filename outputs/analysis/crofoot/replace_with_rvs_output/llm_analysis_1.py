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
    """
    Prepare variables for modeling the probability that the focal group wins an intergroup contest.

    Input columns required (must be present in df):
      - win, n_focal, n_other, dist_focal, dist_other, m_focal, m_other, f_focal, f_other, dyad

    New columns added to the returned dataframe (these are used in the model):
      - size_diff: numeric, n_focal - n_other
      - size_ratio: numeric, n_focal / n_other (auxiliary)
      - male_diff: m_focal - m_other
      - female_diff: f_focal - f_other
      - dist_adv: dist_other - dist_focal (positive => focal closer to its center)
      - ContestLocation: categorical with values 'FocalHome', 'Neutral', 'OtherHome' (moderator)

    Rows with missing values in required variables are dropped.
    """
    df = df.copy()

    # Drop rows with missing essential fields for the planned model
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    df = df.dropna(subset=required)

    # Ensure numeric types where appropriate
    for col in ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other']:
        # try to coerce to numeric; errors -> NaN which will be dropped above
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derived predictors
    df['size_diff'] = df['n_focal'] - df['n_other']
    # also keep ratio for diagnostics/robustness checks
    # guard division by zero (n_other should not be zero in this dataset but be safe)
    df['size_ratio'] = df['n_focal'] / df['n_other'].replace({0: np.nan})

    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Distance advantage: positive means focal is closer to its home-range center than other
    df['dist_adv'] = df['dist_other'] - df['dist_focal']

    # Create a categorical ContestLocation moderator based on which group is closer to its center
    # Use a small distance threshold to define 'Neutral' when distances are similar
    threshold_meters = 20  # adjustable threshold for 'neutral' zone
    df['ContestLocation'] = pd.Series(np.where(
        df['dist_focal'] + threshold_meters < df['dist_other'],
        'FocalHome',
        np.where(df['dist_other'] + threshold_meters < df['dist_focal'], 'OtherHome', 'Neutral')
    ), index=df.index)

    df['ContestLocation'] = pd.Categorical(df['ContestLocation'], categories=['FocalHome', 'Neutral', 'OtherHome'])

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Return only columns needed for modeling (keep originals that might be useful)
    # We'll keep original group sizes and distances plus derived columns and dyad
    keep_cols = list(df.columns)  # keep full frame by default; model will select necessary columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting probability focal group wins (win = 1) from relative group size
    and contest location, with an interaction between size_diff and ContestLocation, and control variables.

    Model: win ~ size_diff * ContestLocation + male_diff + female_diff + dist_adv

    Clusters standard errors by dyad to account for non-independence within dyads.

    Returns:
      - results: statsmodels results object with cluster-robust covariance (if available)
    """
    import statsmodels.formula.api as smf

    # Ensure the expected columns exist
    required = ['win', 'size_diff', 'ContestLocation', 'male_diff', 'female_diff', 'dist_adv', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula with interaction between size_diff and ContestLocation (ContestLocation acts as moderator)
    formula = 'win ~ size_diff * ContestLocation + male_diff + female_diff + dist_adv'

    # Fit logistic regression (maximum likelihood)
    model = smf.logit(formula, data=df)
    fitted = model.fit(disp=False)

    # Attempt cluster-robust covariance on dyad. If it fails, return the plain fitted model.
    try:
        results = fitted.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If clustering fails for any reason, return the standard fitted object
        results = fitted

    return results


