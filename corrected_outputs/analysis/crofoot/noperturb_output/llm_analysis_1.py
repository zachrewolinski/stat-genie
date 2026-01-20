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
    Transform the raw capuchin contest data to create the variables needed for modeling.

    Outputs (added columns used in the model):
    - size_diff: n_focal - n_other
    - LocAdv: dist_other - dist_focal (positive means focal is closer to its home center)
    - male_diff: m_focal - m_other
    - female_diff: f_focal - f_other
    - size_diff_z, LocAdv_z, male_diff_z, female_diff_z: z-scored versions (mean 0, sd 1)
    - FocalHomeAdv: binary indicator (1 if dist_focal < dist_other else 0)

    Also coerces numeric columns, drops rows with missing critical values.
    """
    df = df.copy()

    # Ensure critical columns are numeric (coerce invalid entries to NaN)
    numeric_cols = [
        'win', 'dist_focal', 'dist_other',
        'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other',
        'dyad'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any of the required columns
    required = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Compute difference measures
    df['size_diff'] = df['n_focal'] - df['n_other']
    df['LocAdv'] = df['dist_other'] - df['dist_focal']
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Create a binary indicator for whether the contest was closer to the focal group's home center
    df['FocalHomeAdv'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize (z-score) the continuous predictors used in the model.
    # Use population std (ddof=0). If std is zero (constant column), set z to 0.
    def zscore_safe(x: pd.Series) -> pd.Series:
        std = x.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / std

    df['size_diff_z'] = zscore_safe(df['size_diff'])
    df['LocAdv_z'] = zscore_safe(df['LocAdv'])
    df['male_diff_z'] = zscore_safe(df['male_diff'])
    df['female_diff_z'] = zscore_safe(df['female_diff'])

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Return transformed dataframe with all new columns present
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic (binomial GLM) predicting the probability that the focal group wins.

    Model specification:
    win ~ size_diff_z * LocAdv_z + male_diff_z + female_diff_z + C(dyad)

    - size_diff_z: standardized relative group size (focal - other)
    - LocAdv_z: standardized location advantage (dist_other - dist_focal)
    - Interaction tests whether the effect of relative size depends on location advantage
    - male_diff_z, female_diff_z: standardized compositional controls
    - C(dyad): dyad fixed effects to control for pair-specific heterogeneity

    Returns the fitted statsmodels result object (GLMResults).
    """
    import statsmodels.formula.api as smf

    # Check that required columns exist
    required_cols = ['win', 'size_diff_z', 'LocAdv_z', 'male_diff_z', 'female_diff_z', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit binomial GLM (logistic regression)
    formula = 'win ~ size_diff_z * LocAdv_z + male_diff_z + female_diff_z + C(dyad)'
    results = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    return results


