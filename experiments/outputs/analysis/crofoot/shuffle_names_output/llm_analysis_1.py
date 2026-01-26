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
    Transform raw capuchin contest data to variables used in modeling.

    Assumptions based on provided schema descriptions:
      - 'dyad' is the binary outcome (1 if focal won, 0 if other won).
      - 'f_other' and 'f_focal' are described as "Number of individuals in focal group" and
        "Number of individuals in other group" respectively (descriptions in schema are inconsistent
        with column names; here we follow the schema descriptions rather than rely solely on column names).
      - 'win' = distance (m) of focal group from its home-range center.
      - 'm_focal' = distance (m) of other group from its home-range center.

    Produces the following new columns (which are used in the statistical model):
      - focal_total, other_total, size_diff, size_ratio, size_ratio_z
      - location_advantage = other_distance - focal_distance (positive if contest nearer focal home)
      - location_advantage_z, focal_home (binary indicator)
      - male_diff

    Any rows with impossible or missing values needed for the model are dropped.
    """
    df = df.copy()

    # Ensure dyad is integer 0/1
    df['dyad'] = df['dyad'].astype(float)
    df = df.dropna(subset=['dyad'])
    df['dyad'] = df['dyad'].astype(int)

    # Create total group-size variables using the columns indicated in the schema descriptions
    # f_other -> described as number of individuals in focal group
    # f_focal -> described as number of individuals in other group
    df['focal_total'] = pd.to_numeric(df['f_other'], errors='coerce')
    df['other_total'] = pd.to_numeric(df['f_focal'], errors='coerce')

    # If either total is missing or zero, mark as missing (can't compute ratio)
    df.loc[df['other_total'] == 0, 'other_total'] = np.nan

    # Size difference and ratio
    df['size_diff'] = df['focal_total'] - df['other_total']
    df['size_ratio'] = df['focal_total'] / df['other_total']

    # Male counts: n_focal described as number of males in focal group; 'other' described as number of males in other group
    df['n_focal'] = pd.to_numeric(df['n_focal'], errors='coerce')
    df['other'] = pd.to_numeric(df['other'], errors='coerce')
    df['male_diff'] = df['n_focal'] - df['other']

    # Contest location: 'win' is focal distance from its home center; 'm_focal' is other group's distance
    df['focal_dist_from_home'] = pd.to_numeric(df['win'], errors='coerce')
    df['other_dist_from_home'] = pd.to_numeric(df['m_focal'], errors='coerce')

    # location_advantage: positive means contest is relatively closer to focal group's center than the other group's center
    df['location_advantage'] = df['other_dist_from_home'] - df['focal_dist_from_home']

    # Binary indicator: contest is closer to focal home (1) vs not (0)
    df['focal_home'] = (df['location_advantage'] > 0).astype(int)

    # Drop rows missing the key predictors or outcome
    df = df.dropna(subset=['focal_total', 'other_total', 'location_advantage', 'dyad'])

    # Standardize continuous predictors (z-score) for model stability and interpretation
    for col in ['size_ratio', 'location_advantage', 'size_diff', 'male_diff']:
        if col in df.columns:
            # compute z-score; if constant or all NaN, set to NaN
            vals = pd.to_numeric(df[col], errors='coerce')
            if vals.dropna().shape[0] > 1 and vals.std() > 0:
                df[col + '_z'] = (vals - vals.mean()) / vals.std()
            else:
                df[col + '_z'] = np.nan

    # Keep only columns relevant for modeling and diagnostics (plus original ids)
    keep_cols = [
        'dyad',
        'focal_total', 'other_total', 'size_diff', 'size_ratio', 'size_ratio_z',
        'location_advantage', 'location_advantage_z', 'focal_home',
        'male_diff', 'n_focal', 'other',
        'm_other',  # dyad / encounter id (categorical control)
        'focal_dist_from_home', 'other_dist_from_home'
    ]

    # Some of these may not exist if conversion failed; filter
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting focal-group victory (dyad) from
    relative group size, contest location, their interaction, and controls.

    Model specification (primary):
      dyad ~ size_ratio_z * location_advantage_z + male_diff + C(m_other)

    - size_ratio_z: standardized focal/other group size ratio
    - location_advantage_z: standardized continuous contest-location advantage (positive -> closer to focal home)
    - male_diff: difference in number of adult males (focal - other)
    - C(m_other): categorical dyad/encounter ID as a fixed-effect control

    Returns the fitted model object (statsmodels result) so the caller can inspect coefficients, CIs, etc.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Make a copy to avoid modifying input
    df = df.copy()

    # Ensure necessary columns exist
    required = ['dyad', 'size_ratio_z', 'location_advantage_z', 'male_diff', 'm_other']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop rows with missing predictors
    model_df = df.dropna(subset=['dyad', 'size_ratio_z', 'location_advantage_z', 'male_diff'])

    # Fit binomial GLM with robust (HC3) standard errors
    formula = 'dyad ~ size_ratio_z * location_advantage_z + male_diff + C(m_other)'
    glm_binom = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial())
    results = glm_binom.fit()

    # Attach robust covariance (HC3) - supply clamped try/except in case of issues
    try:
        results_robust = results.get_robustcov_results(cov_type='HC3')
    except Exception:
        results_robust = results

    # Return the robust-results object (has .summary(), .params, .bse, .conf_int(), etc.)
    return results_robust


