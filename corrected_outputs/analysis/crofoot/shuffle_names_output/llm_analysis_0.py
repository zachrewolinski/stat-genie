from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataframe into the analytic dataframe.

    Derivations (new columns):
    - focal_size: number of individuals in the focal group (from column 'f_other', described as "Number of individuals in focal group").
    - other_size: number of individuals in the other group (from column 'f_focal').
    - size_diff: focal_size - other_size.
    - size_ratio: focal_size / other_size (NaN if other_size == 0).
    - focal_dist_home: distance of focal group from center of its home range (from column 'win').
    - other_dist_home: distance of other group from center of its home range (from column 'm_focal').
    - Location: 'FocalHome' if focal_dist_home < other_dist_home, 'OtherHome' if other_dist_home < focal_dist_home, otherwise 'Neutral'.
    - focal_males / other_males: number of males in focal / other groups (from 'n_focal' and 'other').
    - focal_females / other_females: number of females in focal / other groups (from 'dist_focal' and 'focal').
    - Win: dependent variable copied from 'dyad' (1 if focal won, 0 otherwise).

    The function drops rows with missing values in the core columns used for the model.
    """
    # Work on a copy
    df = df.copy()

    # Map sizes (note: original dataset has inconsistent naming in descriptions; these mappings follow the provided field descriptions)
    df['focal_size'] = df['f_other']
    df['other_size'] = df['f_focal']

    # Basic checks and drop rows with missing values in required columns
    required = ['dyad', 'focal_size', 'other_size', 'win', 'm_focal', 'n_focal', 'other']
    df = df.dropna(subset=required)

    # Create outcome column
    # 'dyad' is 1 if focal won, 0 otherwise
    df['Win'] = df['dyad'].astype(int)

    # Size-based predictors
    df['size_diff'] = df['focal_size'] - df['other_size']
    # avoid division by zero
    df['size_ratio'] = df['focal_size'] / df['other_size']
    df.loc[~np.isfinite(df['size_ratio']), 'size_ratio'] = np.nan

    # Location predictors: distances from home-range centers
    df['focal_dist_home'] = df['win']        # described as distance of focal group from center of its home range
    df['other_dist_home'] = df['m_focal']    # described as distance of other group from center of its home range

    # Derive categorical location: closer to focal home, closer to other home, or neutral
    def classify_location(row):
        if pd.isnull(row['focal_dist_home']) or pd.isnull(row['other_dist_home']):
            return np.nan
        if row['focal_dist_home'] < row['other_dist_home']:
            return 'FocalHome'
        elif row['other_dist_home'] < row['focal_dist_home']:
            return 'OtherHome'
        else:
            return 'Neutral'

    df['Location'] = df.apply(classify_location, axis=1)

    # Counts of males/females
    df['focal_males'] = df['n_focal']
    df['other_males'] = df['other']
    df['focal_females'] = df['dist_focal']
    df['other_females'] = df['focal']

    # Final drop: ensure no missing values in model columns
    model_cols = ['Win', 'size_diff', 'size_ratio', 'Location', 'focal_males', 'other_males']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (GLM with binomial family) predicting probability that the focal group wins (Win)
    from relative group size (size_diff and size_ratio), contest location (Location), and controls focal_males
    and other_males. Include interaction terms between size_diff and location to test whether location moderates
    the effect of relative group size.

    Returns the fitted GLMResults object.
    """
    import statsmodels.api as sm

    # Prepare design matrix
    X = df[['size_diff', 'size_ratio', 'focal_males', 'other_males']].copy()

    # One-hot encode Location, drop first to avoid multicollinearity
    loc_dummies = pd.get_dummies(df['Location'], prefix='Location', drop_first=True)
    X = pd.concat([X, loc_dummies], axis=1)

    # Add interaction terms between size_diff and each location dummy (if any)
    for col in loc_dummies.columns:
        X[f'{col}_x_size_diff'] = X[col] * X['size_diff']

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    y = df['Win'].astype(int)

    # Fit GLM with binomial family (logistic regression)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    results = model.fit()

    # Return the fitted results object (user can call results.summary())
    return results


