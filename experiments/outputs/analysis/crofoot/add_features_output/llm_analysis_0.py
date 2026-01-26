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
    Transform the raw capuchin contest dataframe into an analysis-ready dataframe.

    Produces the following columns used in the model:
      - win: dependent binary outcome (keeps original)
      - size_diff_z: standardized (n_focal - n_other)
      - loc_adv_z: standardized (dist_other - dist_focal) ; positive => focal closer to its center
      - homefield: binary indicator (1 if dist_focal < dist_other else 0)
      - male_diff_z: standardized (m_focal - m_other)
      - female_diff_z: standardized (f_focal - f_other)
      - dyad: categorical dyad id (kept for fixed effects / clustering)

    The function drops rows with missing values in required columns.
    """

    df = df.copy()

    # Columns required for analysis
    required_cols = [
        'win', 'dist_focal', 'dist_other',
        'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other',
        'dyad'
    ]

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Compute difference variables (raw)
    df['size_diff'] = df['n_focal'] - df['n_other']
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Location advantage: positive when focal is relatively closer to its home center
    df['loc_adv'] = df['dist_other'] - df['dist_focal']

    # Homefield indicator: 1 if focal is closer to its home center than the other group
    df['homefield'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for stable estimates.
    for col in ['size_diff', 'male_diff', 'female_diff', 'loc_adv']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isclose(std, 0):
            # If zero variance (unlikely), create zero column to avoid division by zero
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Ensure dyad is a categorical variable (keeps it for fixed effects)
    df['dyad'] = df['dyad'].astype('category')

    # Keep only the columns needed for modeling and return a copy
    final_cols = [
        'win',
        'size_diff_z',
        'loc_adv_z',
        'homefield',
        'male_diff_z',
        'female_diff_z',
        'dyad'
    ]

    return df[final_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting the probability that the focal group wins.

    Model specification:
      win ~ size_diff_z * homefield + loc_adv_z + male_diff_z + female_diff_z + C(dyad)

    - Tests main effects of relative group size (size_diff_z) and location advantage (loc_adv_z),
      and whether the effect of relative size varies when the focal group is 'homefield' (interaction).
    - Includes dyad fixed effects via C(dyad) to account for pair-level idiosyncrasies.

    Returns the fitted statsmodels GLMResults object.
    """

    import statsmodels.formula.api as smf

    # Ensure dyad is categorical inside the modeling environment
    df = df.copy()
    df['dyad'] = df['dyad'].astype('category')

    formula = 'win ~ size_diff_z * homefield + loc_adv_z + male_diff_z + female_diff_z + C(dyad)'

    # Fit binomial GLM (logistic regression)
    model_res = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model object (caller can inspect .summary(), .params, etc.)
    return model_res


