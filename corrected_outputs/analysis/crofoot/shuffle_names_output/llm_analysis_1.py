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
    Transform raw dataset into a modeling-ready dataframe.

    Key steps:
    - Drop rows missing any columns required for the core variables.
    - Create WinFocal (binary outcome) from 'dyad'.
    - Compute RelGroupSize = f_focal / (f_focal + f_other).
    - Compute SizeAdv = f_focal - f_other and MaleAdv = n_focal - other.
    - Derive simple contest-location categorical variable 'Location' using focal and other distances.
    - Expose DyadID for clustering.

    Assumptions (based on available fields):
    - 'f_focal' and 'f_other' represent total group sizes for focal and other groups respectively.
    - 'win' and 'm_focal' are interpreted here as continuous distance measures (meters) of focal and other groups from their respective home-range centers; we call them FocalDist and OtherDist.
    - 'dyad' is 1 when the focal group won and 0 otherwise.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()

    # Required columns for analysis
    required_cols = ['dyad', 'f_focal', 'f_other', 'win', 'm_focal', 'n_focal', 'other', 'm_other']
    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Outcome: whether focal group won
    df['WinFocal'] = df['dyad'].astype(int)

    # Distances: interpret 'win' as focal group's distance and 'm_focal' as other group's distance (both in meters)
    df['FocalDist'] = pd.to_numeric(df['win'], errors='coerce')
    df['OtherDist'] = pd.to_numeric(df['m_focal'], errors='coerce')

    # Relative group size (proportion). Protect against division by zero.
    total_size = (pd.to_numeric(df['f_focal'], errors='coerce') + pd.to_numeric(df['f_other'], errors='coerce'))
    df['RelGroupSize'] = pd.to_numeric(df['f_focal'], errors='coerce') / total_size
    df.loc[total_size == 0, 'RelGroupSize'] = np.nan

    # Absolute size advantage and male advantage
    df['SizeAdv'] = pd.to_numeric(df['f_focal'], errors='coerce') - pd.to_numeric(df['f_other'], errors='coerce')
    df['MaleAdv'] = pd.to_numeric(df['n_focal'], errors='coerce') - pd.to_numeric(df['other'], errors='coerce')

    # Dyad identifier (used for clustering standard errors)
    df['DyadID'] = df['m_other'].astype(pd.Int64Dtype())

    # Derive Location categorical variable. Threshold for 'Neutral' set to 50 meters (can be adjusted).
    # If focal is closer to its home center than the other group -> 'Home'. If further -> 'Away'.
    # If distance difference is <= 50 meters -> 'Neutral'.
    df['DistDiff'] = df['OtherDist'] - df['FocalDist']
    threshold = 50.0
    df['Location'] = np.where(df['DistDiff'].abs() <= threshold, 'Neutral', np.where(df['FocalDist'] < df['OtherDist'], 'Home', 'Away'))

    # Keep the transformed columns needed for modeling (plus a few diagnostics)
    model_cols = ['WinFocal', 'RelGroupSize', 'Location', 'MaleAdv', 'SizeAdv', 'DyadID', 'FocalDist', 'OtherDist']

    # Return dataframe with at least the model_cols present; keep other columns as well for inspection
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic (binomial) regression to estimate how relative group size and contest location
    influence the probability that the focal group wins. Includes interaction between RelGroupSize
    and Location to test whether size effects differ by location. Controls include MaleAdv and SizeAdv.

    Uses clustered (dyad-level) robust standard errors to account for non-independence of repeated
    interactions within dyads.

    Returns the fitted model with cluster-robust covariance if available.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Drop rows with missing values in the model variables
    model_df = df.dropna(subset=['WinFocal', 'RelGroupSize', 'Location', 'MaleAdv', 'SizeAdv', 'DyadID']).copy()

    # Ensure proper data types
    model_df['WinFocal'] = model_df['WinFocal'].astype(int)
    model_df['RelGroupSize'] = pd.to_numeric(model_df['RelGroupSize'], errors='coerce')
    model_df['MaleAdv'] = pd.to_numeric(model_df['MaleAdv'], errors='coerce')
    model_df['SizeAdv'] = pd.to_numeric(model_df['SizeAdv'], errors='coerce')
    model_df['DyadID'] = model_df['DyadID'].astype(int)
    model_df['Location'] = model_df['Location'].astype('category')

    # Specify formula with interaction between relative size and location (categorical)
    formula = 'WinFocal ~ RelGroupSize * C(Location) + MaleAdv + SizeAdv'

    # Fit Binomial GLM (logistic regression)
    glm_binom = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Obtain cluster-robust standard errors at the DyadID level
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=model_df['DyadID'])
    except Exception:
        # If clustering fails for any reason, return the original result with a warning
        print('Warning: clustered robust covariance estimation failed; returning default fit.')
        res_cluster = res

    # Print and return the clustered result object (or the default fit if clustering not available)
    print(res_cluster.summary())
    return res_cluster


