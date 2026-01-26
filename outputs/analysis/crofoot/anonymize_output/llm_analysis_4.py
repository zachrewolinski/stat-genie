from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/anonymize_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling dataframe with clearly named columns and derived predictors.

    Final columns created/kept (used in modeling):
      - focal_id, other_id, dyad_id: integer ids
      - focal_win: binary outcome (0/1)
      - focal_dist, other_dist: distances (meters)
      - focal_size, other_size: group sizes (counts)
      - focal_males, other_males, focal_females, other_females: sex composition counts
      - size_ratio: focal_size / other_size
      - size_diff: focal_size - other_size
      - male_diff, female_diff: focal - other
      - size_ratio_z, male_diff_z, female_diff_z: z-scored versions
      - LocationFocal: 1 if contest is closer to focal group's center than other group's center, else 0
    """
    df = df.copy()

    # Rename raw columns to meaningful names
    rename_map = {
        'feature1': 'focal_id',
        'feature2': 'other_id',
        'feature3': 'dyad_id',
        'feature4': 'focal_win',
        'feature5': 'focal_dist',
        'feature6': 'other_dist',
        'feature7': 'focal_size',
        'feature8': 'other_size',
        'feature9': 'focal_males',
        'feature10': 'other_males',
        'feature11': 'focal_females',
        'feature12': 'other_females'
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric types
    num_cols = ['focal_id','other_id','dyad_id','focal_win',
                'focal_dist','other_dist','focal_size','other_size',
                'focal_males','other_males','focal_females','other_females']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing critical fields
    df = df.dropna(subset=['focal_win','focal_size','other_size','focal_dist','other_dist'])

    # Convert focal_win to integer 0/1 (may already be 0/1)
    df['focal_win'] = df['focal_win'].astype(int)

    # Derived predictors
    # Relative size (ratio) and difference
    # Avoid division by zero: drop rows where other_size == 0
    df = df[df['other_size'] != 0]
    df['size_ratio'] = df['focal_size'] / df['other_size']
    df['size_diff'] = df['focal_size'] - df['other_size']

    # Sex composition differences
    df['male_diff'] = df['focal_males'] - df['other_males']
    df['female_diff'] = df['focal_females'] - df['other_females']

    # Location indicator: contest is nearer focal group's center than other group's center
    # If focal_dist <= other_dist -> contest is in/near focal territory
    df['LocationFocal'] = (df['focal_dist'] <= df['other_dist']).astype(int)

    # Standardize (z-score) continuous predictors used in models for interpretability
    # Use sample std (ddof=1)
    for col in ['size_ratio','male_diff','female_diff']:
        mean = df[col].mean()
        std = df[col].std(ddof=1)
        if std == 0 or np.isnan(std):
            # If no variation, create zero column
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Keep only columns needed for modeling (and identifiers)
    keep_cols = [
        'focal_id','other_id','dyad_id','focal_win',
        'focal_dist','other_dist','focal_size','other_size',
        'focal_males','other_males','focal_females','other_females',
        'size_ratio','size_diff','male_diff','female_diff',
        'size_ratio_z','male_diff_z','female_diff_z','LocationFocal'
    ]

    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    # Ensure dyad_id is integer for clustering purposes
    df['dyad_id'] = df['dyad_id'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting the probability that the focal group wins.

    Model formula (in words): focal_win ~ size_ratio_z * LocationFocal + male_diff_z + female_diff_z
    - Tests whether relative size advantage has a different effect when the contest occurs in/near focal territory
    - Uses cluster-robust standard errors clustered by dyad_id to account for non-independence of observations within dyads

    Returns:
      - results_robust: the GLMResults object with cluster-robust covariance
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['focal_win','size_ratio_z','LocationFocal','male_diff_z','female_diff_z','dyad_id']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Design matrix
    # We'll include: intercept, size_ratio_z, LocationFocal, interaction, male_diff_z, female_diff_z
    df['interaction'] = df['size_ratio_z'] * df['LocationFocal']

    X = df[['size_ratio_z','LocationFocal','interaction','male_diff_z','female_diff_z']].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['focal_win']

    # Fit binomial GLM
    model_glm = sm.GLM(y, X, family=sm.families.Binomial())
    results = model_glm.fit()

    # Get cluster-robust covariance clustered by dyad_id
    # This returns a results object with adjusted covariances and tvalues
    try:
        results_robust = results.get_robustcov_results(cov_type='cluster', groups=df['dyad_id'])
    except Exception:
        # If cluster robust fails for any reason, fall back to the original results
        results_robust = results

    # Print summary for user inspection (optional)
    print(results_robust.summary())

    return results_robust


