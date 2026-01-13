from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize and derive variables needed for modeling intergroup contest outcomes.

    Assumptions/notes about original columns (based on provided schema descriptions):
    - 'f_other' contains the total number of individuals in the focal group.
    - 'f_focal' contains the total number of individuals in the other group.
    - 'win' contains the focal group's distance (meters) from center of its home range.
    - 'm_focal' contains the other group's distance (meters) from center of its home range.
    - 'n_focal' contains number of males in focal group.
    - 'other' contains number of males in other group.
    - 'dist_focal' contains number of females in focal group (kept if needed, but not used directly here).
    - 'focal' contains number of females in other group.
    - 'dyad' is the binary contest outcome (1 if focal won, 0 if other won).

    The function will create clear named columns used in the model: focal_total, other_total, rel_size_ratio,
    rel_size_diff, focal_dist, other_dist, dist_diff, contest_location, focal_males, other_males.
    """
    df = df.copy()

    # Ensure dyad is binary integer
    df['dyad'] = df['dyad'].astype(int)

    # Map / derive clear columns (use descriptions from schema)
    # Total group sizes
    df['focal_total'] = df['f_other']
    df['other_total'] = df['f_focal']

    # Male counts
    df['focal_males'] = df['n_focal']
    df['other_males'] = df['other']

    # Female counts (kept if needed later)
    df['focal_females'] = df['dist_focal']
    df['other_females'] = df['focal']

    # Distances from home-range centers to contest location (per schema descriptions)
    df['focal_dist'] = df['win']
    df['other_dist'] = df['m_focal']

    # Relative size (ratio and difference)
    # Avoid division by zero by replacing zeros (unlikely given schema) with np.nan
    df['other_total'] = df['other_total'].replace({0: np.nan})
    df['rel_size_ratio'] = df['focal_total'] / df['other_total']
    df['rel_size_diff'] = df['focal_total'] - df['other_total']

    # Distance difference (other_dist - focal_dist): positive means other is farther from its center than focal
    df['dist_diff'] = df['other_dist'] - df['focal_dist']

    # Categorical contest location: which group is nearer to the contest location
    # If focal_dist < other_dist => focal group's home is nearer to contest => 'FocalNear'
    # If focal_dist > other_dist => other group's home is nearer => 'OtherNear'
    # If equal (rare) => 'Neutral'
    df['contest_location'] = np.where(df['focal_dist'] < df['other_dist'],
                                      'FocalNear',
                                      np.where(df['focal_dist'] > df['other_dist'], 'OtherNear', 'Neutral'))

    # Drop rows missing any required modeling columns
    required = [
        'dyad',
        'rel_size_ratio',
        'rel_size_diff',
        'contest_location',
        'focal_males',
        'other_males',
        'dist_diff'
    ]
    df = df.dropna(subset=required)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM / Logit) predicting the probability that the focal group wins
    (dyad == 1) as a function of relative size, contest location, their interaction, and controls.

    Model specification (final):
      logit(P(dyad=1)) = const + rel_size_ratio + contest_location_dummies + rel_size_ratio:contest_location_dummies
                          + focal_males + other_males + dist_diff + rel_size_diff

    Interaction terms test whether the effect of relative size differs depending on contest location.
    """
    df2 = df.copy()

    # Prepare design matrix
    # Create dummies for contest_location; drop_first=True to use one level as reference
    loc_dummies = pd.get_dummies(df2['contest_location'], prefix='contest_loc', drop_first=True)

    # Base predictors and controls
    X = pd.concat([
        df2[['rel_size_ratio', 'focal_males', 'other_males', 'dist_diff', 'rel_size_diff']].astype(float),
        loc_dummies
    ], axis=1)

    # Add interaction terms: rel_size_ratio * each location dummy
    for col in loc_dummies.columns:
        X[f'rel_size_ratio:{col}'] = X['rel_size_ratio'] * X[col]

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Response
    y = df2['dyad'].astype(int)

    # Fit logistic regression (Logit). Use try/except to fall back to GLM if convergence issues.
    try:
        model_res = sm.Logit(y, X).fit(disp=False)
    except Exception:
        # fall back to GLM binomial with default freq weights if Logit fails
        model_res = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    return model_res


