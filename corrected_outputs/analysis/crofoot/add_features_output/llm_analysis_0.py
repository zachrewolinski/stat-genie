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
    Transform the raw ARTS contest dataframe into the modeling dataframe.

    Produces the following new columns used in the model:
      - size_diff: n_focal - n_other
      - dist_diff: dist_other - dist_focal  (positive => contest relatively closer to focal's home)
      - male_diff: m_focal - m_other
      - female_diff: f_focal - f_other
      - *_z: z-scored versions of the above continuous predictors
      - dyad: categorical dyad indicator

    Drops rows with missing values in any of the columns required for the model.
    """
    df = df.copy()

    # Required input columns
    required_cols = [
        'win',
        'n_focal', 'n_other',
        'dist_focal', 'dist_other',
        'm_focal', 'm_other',
        'f_focal', 'f_other',
        'dyad'
    ]

    # Ensure these columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Coerce numeric columns to numeric dtype where appropriate
    num_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing any required numeric values
    df = df.dropna(subset=num_cols + ['dyad'])

    # Derived predictors
    df['size_diff'] = df['n_focal'] - df['n_other']
    # Positive dist_diff means other is farther from its home center than focal is from its home center => advantage to focal
    df['dist_diff'] = df['dist_other'] - df['dist_focal']
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Standardize (z-score) continuous predictors for better interpretability / numerical stability
    for col in ['size_diff', 'dist_diff', 'male_diff', 'female_diff']:
        # Use population SD (ddof=0) to be explicit
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # If std is zero (no variation), create a zero column to avoid division by zero
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Ensure dyad is treated as a categorical variable
    df['dyad'] = df['dyad'].astype('category')

    # Keep only columns necessary for modeling (but do not drop original columns in case user wants them)
    # The model function will rely on: 'win', 'size_diff_z', 'dist_diff_z', 'male_diff_z', 'female_diff_z', 'dyad'

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting the probability that the focal group wins.

    Model formula:
      win ~ size_diff_z * dist_diff_z + male_diff_z + female_diff_z + C(dyad)

    The interaction term tests whether the effect of relative group size on winning probability
    depends on contest location advantage.

    Returns the fitted model result object (statsmodels GLMResults).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['win', 'size_diff_z', 'dist_diff_z', 'male_diff_z', 'female_diff_z', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit a binomial GLM (logistic regression) with dyad as a categorical fixed effect
    formula = 'win ~ size_diff_z * dist_diff_z + male_diff_z + female_diff_z + C(dyad)'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = model.fit()

    # Recommended post-estimation checks (left to user): examine results.summary(), check predicted probabilities,
    # consider clustered standard errors by dyad or focal if dependence is a concern.

    return results


