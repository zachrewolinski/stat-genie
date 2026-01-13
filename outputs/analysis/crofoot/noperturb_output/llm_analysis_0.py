from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Required original columns
    required_cols = ['win', 'dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'dyad']
    # Drop rows missing any required fields for modeling
    df = df.dropna(subset=required_cols)

    # Ensure types
    df['win'] = df['win'].astype(int)
    for c in ['dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Derived predictors
    # Relative group size (difference and ratio)
    df['size_diff'] = df['n_focal'] - df['n_other']
    df['size_ratio'] = df['n_focal'] / df['n_other'].replace(0, np.nan)

    # Location advantage: positive when contest is closer to focal group's home center
    # (dist_other - dist_focal). Larger positive => greater focal home advantage.
    df['location_adv'] = df['dist_other'] - df['dist_focal']

    # Categorical location for descriptive checks (threshold = 50 meters to define "neutral")
    def _loc_cat(x):
        if x > 50:
            return 'FocalSide'
        elif x < -50:
            return 'OtherSide'
        else:
            return 'Neutral'
    df['contest_location'] = df['location_adv'].apply(_loc_cat)

    # Male composition difference control
    df['m_diff'] = df['m_focal'] - df['m_other']

    # Standardize continuous predictors for model stability/interpretation (z-scores)
    # use population std (ddof=0) to avoid small-sample ddof effects; safe for interpretation
    for col in ['size_diff', 'location_adv', 'm_diff']:
        mu = df[col].mean()
        sigma = df[col].std(ddof=0)
        # If sigma is zero (no variation), create zeros to avoid division by zero
        if sigma == 0 or np.isnan(sigma):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mu) / sigma

    # Final dataframe returned contains original columns plus derived columns used in the model
    # Key columns used by the model: 'win', 'size_diff_z', 'location_adv_z', 'm_diff_z', 'dyad'
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Ensure the transformed columns exist
    needed = ['win', 'size_diff_z', 'location_adv_z', 'm_diff_z', 'dyad']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Specify a logistic regression (GLM with binomial family). Include interaction between
    # relative group size and location advantage to test whether location modulates the
    # effect of relative group size on win probability.
    formula = 'win ~ size_diff_z * location_adv_z + m_diff_z'

    glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial())

    # Fit with cluster-robust standard errors clustered by dyad (accounts for non-independence
    # of contests between the same pair of groups). If dyad clustering is inappropriate,
    # this can be changed to another cluster or to default (non-robust) SEs.
    results = glm.fit(cov_type='cluster', cov_kwds={'groups': df['dyad']})

    # Return the fitted results object (contains coefficients, SEs, summary, predictions, etc.)
    print(results.summary())
    return results


