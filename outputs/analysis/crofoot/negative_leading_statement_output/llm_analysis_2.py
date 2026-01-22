from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/negative_leading_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin intergroup contest dataframe into variables used for modeling.

    New columns added (kept in returned dataframe):
      - size_diff: n_focal - n_other
      - size_ratio: n_focal / n_other
      - male_diff: m_focal - m_other
      - female_diff: f_focal - f_other
      - loc_adv_cont: dist_other - dist_focal  (positive means focal has location advantage)
      - LocAdvBinary: 1 if dist_focal < dist_other else 0
      - size_diff_z, loc_adv_z, male_diff_z, female_diff_z: z-scored versions
      - size_loc_interaction: product of size_diff_z and loc_adv_z

    The function drops rows with missing values in the core variables used below.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Compute group-size and composition differences
    df['size_diff'] = df['n_focal'] - df['n_other']
    # ratio (use float; guard against division by zero though values indicate min 5)
    df['size_ratio'] = df['n_focal'] / df['n_other']
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Location advantage: positive when focal is relatively closer to its own home-range center
    df['loc_adv_cont'] = df['dist_other'] - df['dist_focal']
    # Binary indicator: focal closer to own center than other is to theirs
    df['LocAdvBinary'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Standardize (z-score) key predictors and controls
    def zscore(s):
        return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)

    df['size_diff_z'] = zscore(df['size_diff'])
    df['loc_adv_z'] = zscore(df['loc_adv_cont'])
    df['male_diff_z'] = zscore(df['male_diff'])
    df['female_diff_z'] = zscore(df['female_diff'])

    # Interaction: relative size * location advantage
    df['size_loc_interaction'] = df['size_diff_z'] * df['loc_adv_z']

    # Keep original key columns plus derived columns
    keep_cols = [
        'focal', 'other', 'dyad', 'win',
        'dist_focal', 'dist_other', 'n_focal', 'n_other',
        'm_focal', 'm_other', 'f_focal', 'f_other',
        'size_diff', 'size_ratio', 'male_diff', 'female_diff',
        'loc_adv_cont', 'LocAdvBinary',
        'size_diff_z', 'loc_adv_z', 'male_diff_z', 'female_diff_z',
        'size_loc_interaction'
    ]

    # Some of these columns are guaranteed present; filter to those that exist to avoid KeyError
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (GLM Binomial) predicting probability focal group wins (win==1).

    Primary predictors:
      - size_diff_z (standardized relative group size)
      - loc_adv_z (standardized continuous location advantage)
      - size_loc_interaction (interaction: size_diff_z * loc_adv_z)

    Controls:
      - male_diff_z, female_diff_z

    We compute cluster-robust standard errors clustered by dyad to account for non-independence of contests in the same dyad.

    Returns a dictionary containing the fitted model, clustered-robust results, and odds ratios with 95% CIs computed using clustered SEs.
    """
    results = {}

    # Ensure required columns present
    required_model_cols = [
        'win', 'size_diff_z', 'loc_adv_z', 'size_loc_interaction',
        'male_diff_z', 'female_diff_z', 'dyad'
    ]
    missing = [c for c in required_model_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Response and design matrix
    y = df['win'].astype(float)
    X = df[['size_diff_z', 'loc_adv_z', 'size_loc_interaction', 'male_diff_z', 'female_diff_z']].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit GLM (logistic regression)
    glm_res = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Obtain cluster-robust covariance results clustered by dyad
    try:
        glm_robust = glm_res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # fallback: return non-clustered results if clustering fails
        glm_robust = glm_res

    # Calculate odds ratios and CIs using robust (clustered) SEs when available
    params = glm_robust.params
    bse = glm_robust.bse
    or_vals = np.exp(params)
    ci_lower = np.exp(params - 1.96 * bse)
    ci_upper = np.exp(params + 1.96 * bse)

    or_table = pd.DataFrame({
        'coef': params,
        'se': bse,
        'odds_ratio': or_vals,
        'ci_2.5%': ci_lower,
        'ci_97.5%': ci_upper
    })

    # Add model fit statistics
    try:
        aic = glm_res.aic
    except Exception:
        aic = None

    results['glm_result'] = glm_res
    results['glm_robust'] = glm_robust
    results['odds_ratio_table'] = or_table
    results['aic'] = aic

    # Optional: compute and attach average predicted probability and simple classification metrics
    df = df.copy()
    df['pred_prob'] = glm_robust.predict(X)
    # simple Brier score and mean predicted probability
    results['brier_score'] = np.mean((df['pred_prob'] - df['win']) ** 2)
    results['mean_pred_prob'] = df['pred_prob'].mean()

    # Return dictionary of results for downstream inspection
    return results


