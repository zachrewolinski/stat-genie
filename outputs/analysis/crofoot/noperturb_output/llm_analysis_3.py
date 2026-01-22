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
    """
    Transform original dataframe to produce the columns required for modeling.
    - Drops rows with missing values in key columns used to compute predictors/outcome.
    - Creates relative size and composition differences.
    - Creates distance difference and a binary focal-closer indicator.
    - Standardizes continuous predictors (z-scores) and returns the dataframe with all required columns.
    """
    # copy to avoid modifying original
    df = df.copy()

    # Ensure required columns exist
    required = ['win', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dist_focal', 'dist_other', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing critical values
    df = df.dropna(subset=required)

    # Create relative size: difference (focal - other) and ratio
    df['RelSize_diff'] = df['n_focal'] - df['n_other']
    # ratio (keep for diagnostics; not required for the primary model but helpful)
    # guard division by zero, though n_other >=5 in this dataset
    df['RelSize_ratio'] = df['n_focal'] / df['n_other'].replace({0: np.nan})

    # Composition differences (males, females)
    df['RelMale_diff'] = df['m_focal'] - df['m_other']
    df['RelFem_diff'] = df['f_focal'] - df['f_other']

    # Location: other distance minus focal distance. Positive => focal is closer to its home center than other.
    df['dist_diff'] = df['dist_other'] - df['dist_focal']

    # Binary indicator whether focal is closer to its home-range center than the other
    df['FocalCloser'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize continuous predictors (z-scores). Use population std (ddof=0) for consistency.
    for col in ['RelSize_diff', 'dist_diff']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # if no variance, create zeroed z-score
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Rename standardized columns to match those referenced in the conceptual variables
    # RelSize_diff_z and dist_diff_z are created above

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Keep only columns necessary for modeling plus identifiers for inspection
    keep_cols = ['focal', 'other', 'dyad', 'win',
                 'RelSize_diff', 'RelSize_ratio', 'RelSize_diff_z',
                 'RelMale_diff', 'RelFem_diff',
                 'dist_diff', 'dist_diff_z', 'FocalCloser']

    # Some columns (e.g., focal/other) may have been missing in required earlier; they are optional but helpful
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting probability focal group wins ('win').
    - Primary predictors: RelSize_diff_z (standardized size difference) and dist_diff_z (standardized distance difference).
    - Test for moderation by FocalCloser via an interaction between RelSize_diff_z and FocalCloser.
    - Controls: RelMale_diff and RelFem_diff.
    - Use dyad-clustered robust standard errors to account for non-independence of repeated dyad encounters.

    Returns the fitted results object with clustered robust covariance.
    """
    import statsmodels.formula.api as smf

    # Check that required columns exist
    needed = ['win', 'RelSize_diff_z', 'dist_diff_z', 'FocalCloser', 'RelMale_diff', 'RelFem_diff', 'dyad']
    miss = [c for c in needed if c not in df.columns]
    if len(miss) > 0:
        raise ValueError(f"Missing required columns for modeling: {miss}")

    # Build formula with interaction between size and focal-closer (moderation test)
    formula = 'win ~ RelSize_diff_z * FocalCloser + dist_diff_z + RelMale_diff + RelFem_diff'

    # Fit binomial logistic regression
    model = smf.logit(formula=formula, data=df)
    res = model.fit(disp=False)

    # Obtain cluster-robust standard errors clustered on dyad
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # Fallback: if clustering fails, return regular result but warn the user
        import warnings
        warnings.warn('Cluster-robust SEs failed; returning unclustered results.')
        res_cluster = res

    # Print summary for user inspection and also return the robust results object
    print(res_cluster.summary())

    return res_cluster


