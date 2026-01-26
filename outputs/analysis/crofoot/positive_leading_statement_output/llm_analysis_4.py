from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/positive_leading_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing values in required columns for the analysis
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]
    df = df.dropna(subset=required_cols)

    # Compute relative size measures
    df['rel_size'] = df['n_focal'] - df['n_other']
    # ratio (keeps scale information distinct from difference)
    df['rel_size_ratio'] = df['n_focal'] / df['n_other']

    # Location / home-field advantage measure: positive when focal is closer to its center than other is to its center
    df['location_adv'] = df['dist_other'] - df['dist_focal']
    # Binary flag: focal is inside (closer) vs not
    df['loc_inside'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Sex-composition differences
    df['male_diff'] = df['m_focal'] - df['m_other']
    df['female_diff'] = df['f_focal'] - df['f_other']

    # Total group size control
    df['total_n'] = df['n_focal'] + df['n_other']

    # Standardize continuous predictors (z-scores) for interpretability and numerical stability
    cont_cols = ['rel_size', 'rel_size_ratio', 'location_adv', 'male_diff', 'female_diff', 'total_n']
    for col in cont_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            # If no variation, produce zeros
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Ensure outcome is integer 0/1
    df['win'] = df['win'].astype(int)

    # Keep columns necessary for modelling and inspection
    keep_cols = [
        'win', 'rel_size', 'rel_size_z', 'rel_size_ratio', 'rel_size_ratio_z',
        'location_adv', 'location_adv_z', 'loc_inside',
        'male_diff', 'male_diff_z', 'female_diff', 'female_diff_z',
        'total_n', 'total_n_z', 'dyad', 'focal', 'other'
    ]
    # Some columns (like rel_size_ratio_z) were derived via loop; ensure they exist in df before selecting
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression testing whether relative group size and contest location predict
    probability focal group wins. Uses cluster-robust SEs clustered by dyad.

    Model formula:
      win ~ rel_size_z + location_adv_z + rel_size_z:location_adv_z + male_diff_z + female_diff_z + total_n_z

    The interaction tests whether the effect of relative size depends on location advantage.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present and drop rows with missing predictor values
    model_cols = [
        'win', 'rel_size_z', 'location_adv_z', 'male_diff_z', 'female_diff_z', 'total_n_z', 'dyad'
    ]
    df_model = df.dropna(subset=model_cols).copy()

    # Define formula with interaction
    formula = 'win ~ rel_size_z + location_adv_z + rel_size_z:location_adv_z + male_diff_z + female_diff_z + total_n_z'

    # Fit logistic regression (binomial GLM via logit) and obtain clustered robust standard errors by dyad
    logit_res = smf.logit(formula=formula, data=df_model).fit(disp=False)

    # Cluster-robust covariance by dyad
    try:
        res_cluster = logit_res.get_robustcov_results(cov_type='cluster', groups=df_model['dyad'])
    except Exception:
        # Fallback to default results if clustering fails
        res_cluster = logit_res

    # Print summary (helpful for interactive use)
    print(res_cluster.summary())

    # Return the fitted results object with robust cov if available
    return res_cluster


