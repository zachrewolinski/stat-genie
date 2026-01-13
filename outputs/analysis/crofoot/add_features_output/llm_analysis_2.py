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
    Transform the raw dataset to create the variables used in the statistical model.

    Produced columns used in the model:
    - RelSize_Ratio: n_focal / n_other (continuous)
    - RelSize_Diff: n_focal - n_other (derived, kept for diagnostics)
    - DistDiff: dist_other - dist_focal (continuous signed distance difference)
    - ContestLoc: categorical ('FocalTerritory', 'OtherTerritory', 'Neutral') derived from DistDiff
    - RelMales: m_focal - m_other (control)
    - TotalSize: n_focal + n_other (control)
    - win, dyad remain as in the raw data
    
    Notes: We drop observations with missing values in the core variables required for these derivations.
    A threshold of 50 meters is used to categorize a location as decisively in one group's territory vs neutral.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'dyad']
    df = df.dropna(subset=required_cols)

    # Relative group size: ratio and difference
    # Use float division
    df['RelSize_Ratio'] = df['n_focal'].astype(float) / df['n_other'].astype(float)
    df['RelSize_Diff'] = df['n_focal'] - df['n_other']

    # Distance difference: positive => contest relatively closer to focal's center (other farther)
    df['DistDiff'] = df['dist_other'] - df['dist_focal']

    # Define contest location category using a threshold (50 meters) for a decisive advantage in territory
    # If dist_other - dist_focal > 50 => focal is relatively closer to its center -> 'FocalTerritory'
    # If dist_other - dist_focal < -50 => other is relatively closer to its center -> 'OtherTerritory'
    # Otherwise -> 'Neutral'
    threshold = 50.0
    df['ContestLoc'] = df['DistDiff'].apply(
        lambda x: 'FocalTerritory' if x > threshold else ('OtherTerritory' if x < -threshold else 'Neutral')
    )

    # Composition control: difference in males
    df['RelMales'] = df['m_focal'] - df['m_other']

    # Total size control
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Ensure categorical types for modeling
    df['ContestLoc'] = pd.Categorical(df['ContestLoc'], categories=['FocalTerritory', 'Neutral', 'OtherTerritory'])
    df['dyad'] = df['dyad'].astype('category')

    # Return only the columns necessary for modeling plus a few diagnostics
    keep_cols = ['win', 'RelSize_Ratio', 'RelSize_Diff', 'DistDiff', 'ContestLoc', 'RelMales', 'TotalSize', 'dyad', 'focal', 'other', 'n_focal', 'n_other']
    # If any keep column missing (e.g. focal/other may be absent), fall back to available columns
    available_keep = [c for c in keep_cols if c in df.columns]
    return df[available_keep]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a logistic regression (binomial GLM) predicting the probability that the focal group wins (win == 1)
    as a function of relative group size, contest location, and controls. We include the interaction between
    relative size and contest location to test whether the effect of relative size depends on location.

    Model formula:
    win ~ RelSize_Ratio * C(ContestLoc) + RelMales + TotalSize + C(dyad)

    Returns the fitted model results object (statsmodels GLMResultsWrapper).
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Make a local copy to avoid modifying the input
    dfm = df.copy()

    # Ensure dependent is numeric 0/1
    dfm['win'] = dfm['win'].astype(float)

    # Formula with interaction between relative size and contest location; dyad as fixed effect
    formula = 'win ~ RelSize_Ratio * C(ContestLoc) + RelMales + TotalSize + C(dyad)'

    # Fit binomial GLM (logit link)
    # Use GLM with Binomial family so we can later request robust covariances if desired
    results = smf.glm(formula=formula, data=dfm, family=sm.families.Binomial()).fit()

    # Optionally compute cluster-robust SEs by dyad if there are enough clusters
    # We'll attach a convenience attribute with cluster-robust results if dyad present
    try:
        if 'dyad' in dfm.columns:
            clusters = dfm['dyad']
            # Use cov_type='cluster' with groups=clusters
            robust = results.get_robustcov_results(cov_type='cluster', groups=clusters)
            # store both
            results.cluster_robust = robust
    except Exception:
        # if robust cov computation fails, skip quietly
        results.cluster_robust = None

    return results


