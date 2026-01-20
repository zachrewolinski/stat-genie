from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw ARTS contest dataframe into a dataframe ready for modeling.

    Produces the following new columns used in the model:
    - rel_size_log: np.log(n_focal / n_other)
    - location_advantage: binary indicator (1 if dist_focal < dist_other else 0)
    - male_diff: m_focal - m_other
    - female_diff: f_focal - f_other (kept for possible inspection; not required by model but harmless)
    - total_size: n_focal + n_other
    - dyad: categorical version of the dyad identifier

    Drops rows with missing values in the core variables required for the analysis.
    """
    # Work on a copy
    df = df.copy()

    # Core columns needed for analysis
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Make sure numeric columns are numeric (coerce if necessary)
    numeric_cols = ['n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'win']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop again if coercion produced NaNs in core numeric cols
    df = df.dropna(subset=numeric_cols)

    # Compute relative size as log ratio (safe because group sizes are >0 in this dataset)
    df['rel_size_log'] = np.log(df['n_focal'] / df['n_other'])

    # Binary indicator for whether the focal group is closer to its home-range center than the other group
    df['location_advantage'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Male and female differences
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Total combined group size
    df['total_size'] = df['n_focal'] + df['n_other']

    # Ensure dyad is categorical for downstream creation of dummies / fixed effects
    df['dyad'] = df['dyad'].astype('category')

    # Keep the transformed dataframe with original columns plus new ones
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic (binomial) regression predicting focal-group win (win) from relative size,
    location advantage, their interaction, and control covariates including dyad fixed effects.

    Model specification (shown conceptually):
      logit(P(win=1)) = b0 + b1 * rel_size_log + b2 * location_advantage + b3 * (rel_size_log * location_advantage)
                         + b4 * male_diff + b5 * total_size + dyad_fixed_effects + error

    Returns the fitted statsmodels GLM results object (Binomial family, logit link).
    """
    # Work on a copy
    df = df.copy()

    # Build interaction term
    df['rel_size_log_x_loc'] = df['rel_size_log'] * df['location_advantage']

    # Construct dyad dummies for fixed effects (drop_first to avoid perfect multicollinearity)
    dyad_dummies = pd.get_dummies(df['dyad'], prefix='dyad', drop_first=True)

    # Predictor columns
    predictors = ['rel_size_log', 'location_advantage', 'rel_size_log_x_loc', 'male_diff', 'total_size']

    # Assemble design matrix
    X = pd.concat([df[predictors], dyad_dummies], axis=1)
    X = sm.add_constant(X, has_constant='add')

    # Response
    y = df['win']

    # Fit GLM with binomial family (logit). Use default optimizer.
    model_res = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Return the fitted model object (contains params, summary(), etc.)
    return model_res


