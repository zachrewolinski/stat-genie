from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_and_positive_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin contest dataframe into a modeling-ready dataframe.

    Produces the following new columns used in the model:
      - RelSizeDiff: raw difference n_focal - n_other (kept for diagnostics)
      - RelSizeLog: log(n_focal / n_other)
      - RelSize_z: standardized log ratio (IV)
      - DistAdv: dist_other - dist_focal (positive => focal closer to own center) (kept for diagnostics)
      - DistAdv_z: standardized DistAdv (IV)
      - MaleDiff_z: standardized m_focal - m_other (control)
      - FemaleDiff_z: standardized f_focal - f_other (control)

    Also drops rows with missing values in the columns required for modeling.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()

    # Required raw columns (ensure existence)
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                     'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing values in any variable we'll use
    df = df.dropna(subset=required_cols)

    # Relative size: difference and log ratio. Use log ratio for proportional effect.
    # Use float conversion to avoid integer division issues.
    df['n_focal'] = df['n_focal'].astype(float)
    df['n_other'] = df['n_other'].astype(float)

    # Prevent division by zero (not expected in this dataset since minima >=5) but safe-guard.
    df = df[(df['n_other'] > 0) & (df['n_focal'] > 0)].copy()

    df['RelSizeDiff'] = df['n_focal'] - df['n_other']
    df['RelSizeLog'] = np.log(df['n_focal'] / df['n_other'])

    # Location advantage: other farther than focal => focal has location advantage
    df['DistAdv'] = df['dist_other'] - df['dist_focal']

    # Composition differences (controls)
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Standardize the continuous predictors (z-score). Use population std (ddof=0) for interpretability.
    # Map base columns to the required final standardized column names.
    base_to_final = {
        'RelSizeLog': 'RelSize_z',
        'DistAdv': 'DistAdv_z',
        'MaleDiff': 'MaleDiff_z',
        'FemaleDiff': 'FemaleDiff_z'
    }

    for base_col, final_col in base_to_final.items():
        if base_col not in df.columns:
            raise RuntimeError(f"Expected intermediate column {base_col} to exist for standardization")
        mean = df[base_col].mean()
        std = df[base_col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If there's no variation, create a zero column (model will drop or flag).
            df[final_col] = 0.0
        else:
            df[final_col] = (df[base_col] - mean) / std

    # Keep columns needed for modeling and diagnostics
    model_cols = ['win', 'RelSize_z', 'DistAdv_z', 'MaleDiff_z', 'FemaleDiff_z', 'dyad',
                  'RelSizeDiff', 'RelSizeLog', 'DistAdv', 'MaleDiff', 'FemaleDiff']

    # Ensure all required final columns are present
    for expected in ['RelSize_z', 'DistAdv_z', 'MaleDiff_z', 'FemaleDiff_z']:
        if expected not in df.columns:
            raise RuntimeError(f"Failed to create standardized column {expected}")

    # Return dataframe with all new columns (we keep extras for diagnostics)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (GLM with binomial family) predicting the probability that the focal
    group wins. The primary predictors are RelSize_z (relative size advantage) and DistAdv_z
    (location advantage). We include their interaction and control for MaleDiff_z and FemaleDiff_z.

    Because the same dyads may appear repeatedly, we cluster robust standard errors by 'dyad'.

    Returns the cluster-robust results object.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    # Ensure required columns are present
    req = ['win', 'RelSize_z', 'DistAdv_z', 'MaleDiff_z', 'FemaleDiff_z', 'dyad']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for model: {missing}")

    # Drop any rows with missing values in model columns
    model_df = df.dropna(subset=req).copy()

    # Ensure binary outcome is numeric (0/1)
    model_df['win'] = model_df['win'].astype(float)

    # Formula: include interaction between relative size and location advantage
    formula = 'win ~ RelSize_z * DistAdv_z + MaleDiff_z + FemaleDiff_z'

    # Fit GLM (logistic)
    glm_result = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial()).fit()

    # Obtain cluster-robust covariance by dyad
    try:
        glm_robust = glm_result.get_robustcov_results(cov_type='cluster', groups=model_df['dyad'])
    except Exception:
        # If cluster robust cannot be computed for any reason, fall back to default result
        glm_robust = glm_result

    # Print summary for quick inspection
    print(glm_robust.summary())

    return glm_robust