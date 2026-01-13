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
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required columns check - will raise if missing
    required = ['dyad', 'win', 'm_focal', 'f_other', 'f_focal', 'n_focal', 'other', 'm_other']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for transformation: {missing}")

    # Drop rows with missing values in essential columns
    df = df.dropna(subset=['dyad', 'win', 'm_focal', 'f_other', 'f_focal', 'n_focal', 'other', 'm_other'])

    # Ensure integer/numeric types where appropriate
    numeric_cols = ['dyad', 'win', 'm_focal', 'f_other', 'f_focal', 'n_focal', 'other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    # Derive total group sizes
    # NOTE: field descriptions in the provided schema are inconsistent, so we assume
    #   - f_other contains the number of adult females in the focal group
    #   - f_focal contains the number of adult females in the other group
    #   - n_focal contains the number of adult males in the focal group
    #   - other contains the number of adult males in the other group
    # These assumptions are documented above and used to compute totals below.
    df['total_focal'] = df['f_other'] + df['n_focal']
    df['total_other'] = df['f_focal'] + df['other']

    # Guard against division by zero when computing ratios
    eps = 1e-6
    df['total_other'] = df['total_other'].replace({0: eps})
    df['rel_size_ratio'] = df['total_focal'] / df['total_other']
    df['rel_size_diff'] = df['total_focal'] - df['total_other']

    # Log-transformed relative size (symmetric and interpretable)
    # Add small epsilon to avoid log(0)
    df['log_rel_size'] = np.log(df['rel_size_ratio'].replace(0, eps))

    # Operationalize contest location using distances to group centers.
    # We assume 'win' is the distance (m) from contest location to the focal group's home-range center
    # and 'm_focal' is the distance (m) from contest location to the other group's home-range center
    # (these names / descriptions were inconsistent in the schema; we adopt this mapping below).
    df['dist_to_focal'] = df['win']
    df['dist_to_other'] = df['m_focal']

    # Define contest location categorical variable using a practical threshold.
    # If focal is at least 50 m closer than other -> 'Focal'; if other is >=50 m closer -> 'Other'; else 'Neutral'
    # (50 m threshold chosen as a reasonable buffer given meter-scale distances in the data; adjust if needed.)
    df['ContestLocation'] = 'Neutral'
    df.loc[df['dist_to_focal'] + 50 < df['dist_to_other'], 'ContestLocation'] = 'Focal'
    df.loc[df['dist_to_other'] + 50 < df['dist_to_focal'], 'ContestLocation'] = 'Other'

    # Binary dummy columns for modeling (Neutral is reference)
    df['ContestLoc_Focal'] = (df['ContestLocation'] == 'Focal').astype(int)
    df['ContestLoc_Other'] = (df['ContestLocation'] == 'Other').astype(int)

    # Ensure dyad is integer 0/1
    df['dyad'] = df['dyad'].astype(int)
    df = df[df['dyad'].isin([0, 1])]

    # Keep only columns needed for modeling and diagnostics
    cols_keep = [
        'dyad',
        'total_focal', 'total_other', 'rel_size_ratio', 'rel_size_diff', 'log_rel_size',
        'ContestLocation', 'ContestLoc_Focal', 'ContestLoc_Other',
        'n_focal', 'other', 'm_other',
        'dist_to_focal', 'dist_to_other'
    ]
    # Some of these columns may overlap with original columns; ensure we only return existing ones
    cols_keep = [c for c in cols_keep if c in df.columns]

    return df[cols_keep].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build design matrix for logistic regression (Binomial GLM)
    # Predictors: log_rel_size, Contest location dummies, controls (n_focal, other), and interactions between size and location
    df = df.copy()

    required = ['dyad', 'log_rel_size', 'ContestLoc_Focal', 'ContestLoc_Other', 'n_focal', 'other', 'm_other']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Interaction terms: size x location
    df['log_size_x_Focal'] = df['log_rel_size'] * df['ContestLoc_Focal']
    df['log_size_x_Other'] = df['log_rel_size'] * df['ContestLoc_Other']

    # Predictor matrix
    predictors = [
        'log_rel_size',
        'ContestLoc_Focal', 'ContestLoc_Other',
        'log_size_x_Focal', 'log_size_x_Other',
        'n_focal', 'other'
    ]

    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')
    y = df['dyad']

    # Fit Binomial GLM (logit link)
    model_glm = sm.GLM(y, X, family=sm.families.Binomial())
    res = model_glm.fit()

    # Obtain cluster-robust standard errors clustered by dyad pair identifier (m_other)
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['m_other'])
    except Exception:
        # If clustering fails for any reason, fall back to the default results
        res_clust = res

    # Print a concise summary and return the robust result object
    print(res_clust.summary())
    return res_clust


