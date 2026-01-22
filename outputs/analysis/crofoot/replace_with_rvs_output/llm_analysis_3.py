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
    Transform the raw capuchin contest dataframe into analysis-ready variables.

    Produces the following new columns used in modeling:
    - size_ratio: n_focal / n_other (float)
    - size_diff: n_focal - n_other (int) [kept for exploration but not required by model]
    - FocalHome: binary indicator (1 if dist_focal < dist_other else 0)
    - loc_margin: dist_other - dist_focal (positive when focal is nearer their center)
    - m_diff: m_focal - m_other
    - f_diff: f_focal - f_other
    - n_total: n_focal + n_other

    Drops rows with missing values in required columns.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'win', 'dist_focal', 'dist_other',
        'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other',
        'focal', 'dyad'
    ]

    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    for col in ['dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Re-drop if coercion created NaNs
    df = df.dropna(subset=['dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other'])

    # Derived variables
    df['size_ratio'] = df['n_focal'] / df['n_other']
    df['size_diff'] = df['n_focal'] - df['n_other']

    # Location advantage: focal group is closer to its own home-range center than the other group is to its center
    # (distances are distance from each group's center to the contest location)
    df['FocalHome'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Margin of location advantage (positive means focal is closer)
    df['loc_margin'] = df['dist_other'] - df['dist_focal']

    # Sex-composition differences
    df['m_diff'] = df['m_focal'] - df['m_other']
    df['f_diff'] = df['f_focal'] - df['f_other']

    # Total abundance (control for total number of individuals present)
    df['n_total'] = df['n_focal'] + df['n_other']

    # Ensure identifiers are appropriate types (used for fixed effects / clustering)
    df['focal'] = df['focal'].astype(str)
    df['dyad'] = df['dyad'].astype(str)

    # Keep only relevant columns for modeling (preserve original win column)
    keep_cols = [
        'focal', 'other', 'dyad', 'win',
        'dist_focal', 'dist_other', 'size_ratio', 'size_diff', 'FocalHome', 'loc_margin',
        'n_focal', 'n_other', 'n_total', 'm_focal', 'm_other', 'm_diff', 'f_focal', 'f_other', 'f_diff'
    ]

    # Some datasets may not have 'other' column named; if missing, ignore it
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) regression to estimate how relative group size and contest location
    affect the probability that the focal group wins.

    Model specification (primary):
      win ~ size_ratio + FocalHome + size_ratio:FocalHome + m_diff + f_diff + n_total + C(focal)

    - We include the interaction size_ratio:FocalHome to test whether the effect of relative size
      differs when the focal group has a location/home advantage.
    - C(focal) adds focal-group fixed effects to control for persistent group-level differences.
    - We compute cluster-robust standard errors clustered by dyad to account for repeated
      encounters between the same pair of groups.

    Returns:
      results -> the fitted GLM results with cluster-robust covariance (statsmodels results object)
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure the data contains the columns we created in transform
    required_model_cols = ['win', 'size_ratio', 'FocalHome', 'm_diff', 'f_diff', 'n_total', 'focal', 'dyad']
    missing = [c for c in required_model_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Fit binomial GLM with interaction and focal fixed effects
    formula = 'win ~ size_ratio + FocalHome + size_ratio:FocalHome + m_diff + f_diff + n_total + C(focal)'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    fit = model.fit()

    # Compute cluster-robust standard errors clustered on dyad
    # (some versions of statsmodels require passing the original fit to get_robustcov_results)
    try:
        robust = fit.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # Fallback: return the original fit if clustering fails
        robust = fit

    # Print a concise summary and return the robust results object
    print(robust.summary())
    return robust


