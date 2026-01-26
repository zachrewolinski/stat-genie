from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
from statsmodels.stats.sandwich_covariance import cov_cluster

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Crofoot et al. contest data into analysis-ready columns.

    Inputs (expected raw columns):
      - dyad: 1 if focal won, 0 if other won (outcome)
      - f_other: described in schema as 'Number of individuals in focal group'
      - f_focal: described as 'Number of individuals in other group'
      - win: described as 'Distance in meters of focal group from the center of its home range'
      - m_focal: described as 'Distance in meters of other group from the center of its home range'
      - n_focal: number of males in focal group
      - other: number of males in other group
      - dist_focal: described (in schema) as number of females in focal group
      - focal: described as number of females in other group
      - m_other: dyad ID (kept for clustering / grouping in modeling)

    Outputs (new columns added):
      - Won (0/1 outcome), focal_size, other_size, relative_size, size_ratio,
        focal_dist, other_dist, focal_home_bin (0/1), location (text),
        male_adv, female_adv
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['dyad', 'f_other', 'f_focal', 'win', 'm_focal', 'n_focal', 'other', 'dist_focal', 'focal', 'm_other']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing essential information (outcome, sizes, distances)
    df = df.dropna(subset=['dyad', 'f_other', 'f_focal', 'win', 'm_focal'])

    # Outcome
    df['Won'] = df['dyad'].astype(int)

    # Derive focal and other group sizes (use schema descriptions):
    # NOTE: schema indicates f_other contains number in focal group and f_focal contains number in other group
    df['focal_size'] = pd.to_numeric(df['f_other'], errors='coerce')
    df['other_size'] = pd.to_numeric(df['f_focal'], errors='coerce')

    # Relative size measures
    df['relative_size'] = df['focal_size'] - df['other_size']
    # Ratio (avoid division by zero)
    df['size_ratio'] = df['focal_size'] / df['other_size']
    df.loc[~np.isfinite(df['size_ratio']), 'size_ratio'] = np.nan

    # Distances from home-range centers (schema: 'win' -> focal distance; 'm_focal' -> other distance)
    df['focal_dist'] = pd.to_numeric(df['win'], errors='coerce')
    df['other_dist'] = pd.to_numeric(df['m_focal'], errors='coerce')

    # Location / home advantage: focal_home_bin = 1 when focal is closer to its home-range center than the other group
    df['focal_home_bin'] = (df['focal_dist'] < df['other_dist']).astype(int)
    df['location'] = np.where(df['focal_dist'] < df['other_dist'], 'FocalHome',
                              np.where(df['focal_dist'] > df['other_dist'], 'OtherHome', 'Neutral'))

    # Controls: male and female numerical advantages (schema fields interpreted as follows):
    # n_focal = number of males in focal; other = number of males in other
    df['male_adv'] = pd.to_numeric(df['n_focal'], errors='coerce') - pd.to_numeric(df['other'], errors='coerce')

    # dist_focal and focal columns in schema appear to refer to female counts (description mismatch in schema names)
    df['female_adv'] = pd.to_numeric(df['dist_focal'], errors='coerce') - pd.to_numeric(df['focal'], errors='coerce')

    # Keep the dyad ID (m_other) for clustering in the model
    df['m_other'] = df['m_other']

    # Final: drop rows with any remaining NA in model-relevant columns
    model_cols = ['Won', 'relative_size', 'size_ratio', 'focal_home_bin', 'male_adv', 'female_adv', 'm_other']
    df = df.dropna(subset=model_cols)

    # Convert types to numeric / int where appropriate
    df['Won'] = df['Won'].astype(int)
    df['focal_home_bin'] = df['focal_home_bin'].astype(int)

    # Return full df copy (includes derived cols)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability that the focal group wins an intergroup contest
    from relative group size and contest location (home advantage), controlling for male and female advantages.

    Uses cluster-robust standard errors clustered on 'm_other' (dyad ID) to account for non-independence.

    Returns the robust results object (or a lightweight wrapper providing clustered inference if the
    statsmodels method is not available).
    """
    # Ensure transformed columns exist
    required = ['Won', 'relative_size', 'size_ratio', 'focal_home_bin', 'male_adv', 'female_adv', 'm_other']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build design matrix with interaction between relative_size and focal_home_bin
    X = pd.DataFrame({
        'relative_size': df['relative_size'].astype(float),
        'size_ratio': df['size_ratio'].astype(float),
        'focal_home_bin': df['focal_home_bin'].astype(float),
        'male_adv': df['male_adv'].astype(float),
        'female_adv': df['female_adv'].astype(float)
    })

    # Interaction term: does the effect of relative size differ when focal has home advantage?
    X['relative_size_x_home'] = X['relative_size'] * X['focal_home_bin']

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    y = df['Won'].astype(int)

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)

    # Obtain cluster-robust standard errors clustered by dyad ID (m_other)
    try:
        # Preferred method if available
        clustered_res = res.get_robustcov_results(cov_type='cluster', groups=df['m_other'])
    except AttributeError:
        # Fallback: compute clustered covariance matrix and create a lightweight results wrapper
        cov = cov_cluster(res, df['m_other'])
        bse = np.sqrt(np.diag(cov))
        params = res.params
        # z-statistics using normal approximation
        zvals = params / bse
        pvals = 2 * (1 - scipy.stats.norm.cdf(np.abs(zvals)))
        conf_int = np.column_stack((params - 1.96 * bse, params + 1.96 * bse))

        class ClusteredResultsWrapper:
            def __init__(self, params, bse, zvalues, pvalues, conf_int):
                self.params = params
                self.bse = bse
                self.zvalues = zvalues
                self.pvalues = pvalues
                self.conf_int = conf_int

            def summary(self):
                df_sum = pd.DataFrame({
                    'coef': self.params,
                    'std err': self.bse,
                    'z': self.zvalues,
                    'P>|z|': self.pvalues,
                    '0.025': self.conf_int[:, 0],
                    '0.975': self.conf_int[:, 1]
                })
                return df_sum

        clustered_res = ClusteredResultsWrapper(params, bse, zvals, pvals, conf_int)

    # Print a concise summary (caller can inspect the returned object for details)
    try:
        print(clustered_res.summary())
    except Exception:
        # If the summary can't be printed directly, attempt to display key attributes
        try:
            print("params:\n", clustered_res.params)
            print("std err:\n", getattr(clustered_res, 'bse', None))
        except Exception:
            pass

    return clustered_res