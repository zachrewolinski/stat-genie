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
    Transform the raw capuchin contest dataframe to create the variables used in modeling.

    Outputs (added columns used in model):
      - RelSize_z: standardized (z-scored) RelSize (n_focal - n_other)
      - RelDist_z: standardized (z-scored) RelDist (dist_other - dist_focal)
      - m_diff_z: standardized (z-scored) difference in adult males (m_focal - m_other)
      - f_diff_z: standardized (z-scored) difference in adult females (f_focal - f_other)
      - RelSize_x_RelDist: interaction term (RelSize_z * RelDist_z)
      - dyad: cast to categorical
      - win: outcome (1 if focal won, 0 otherwise)

    Rows with missing values in required columns are dropped.
    """
    df = df.copy()

    # Required columns for the analysis (raw inputs)
    required_cols = [
        'win', 'dist_focal', 'dist_other', 'n_focal', 'n_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Derive raw predictors
    df['RelSize'] = df['n_focal'] - df['n_other']
    # Positive RelDist means focal is closer to its home-range center than the other group
    df['RelDist'] = df['dist_other'] - df['dist_focal']
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Standardize (z-score). Use ddof=0 for population-like standardization; safe for small samples.
    for col in ['RelSize', 'RelDist', 'm_diff', 'f_diff']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # avoid division by zero or NaN std
        if std == 0 or np.isnan(std):
            # create a column of zeros if no variation
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Interaction term between the two standardized main predictors
    df['RelSize_x_RelDist'] = df['RelSize_z'] * df['RelDist_z']

    # Ensure dyad is categorical for fixed-effect style control and for clustering
    df['dyad'] = df['dyad'].astype('category')

    # Keep only columns necessary for modeling plus original outcome (and any helper columns are allowed internally)
    model_cols = [
        'win', 'RelSize_z', 'RelDist_z', 'RelSize_x_RelDist',
        'm_diff_z', 'f_diff_z', 'dyad'
    ]

    # If some of these columns are not present due to earlier issues, keep what we have
    existing = [c for c in model_cols if c in df.columns]
    return df[existing].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting focal win from relative size,
    contest-location advantage, their interaction, and controls. Dyad is included as a
    categorical control (fixed-effect style). Standard errors are clustered by dyad.

    Returns the fitted result object with clustered robust covariance (as produced by the fit).
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns exist
    required = ['win', 'RelSize_z', 'RelDist_z', 'RelSize_x_RelDist', 'm_diff_z', 'f_diff_z', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: main effects + interaction + controls + dyad fixed effects
    formula = 'win ~ RelSize_z + RelDist_z + RelSize_x_RelDist + m_diff_z + f_diff_z + C(dyad)'

    # Fit binomial GLM with clustered robust covariance specified at fit time.
    # Pass dyad groups via cov_kwds so the returned result uses clustered covariance.
    # Using df['dyad'] directly is fine (category or array-like).
    glm_results = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': df['dyad']}
    )

    # Return the fitted result object (with clustered robust covariance)
    return glm_results