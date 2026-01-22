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
    Transform the raw dataset to the analysis-ready dataframe.

    Produces the following analysis columns (all required by the model):
      - win (kept as-is)                    : dependent binary outcome
      - RelSize_log                         : log(n_focal / n_other)
      - LocationAdv                         : dist_other - dist_focal (positive => nearer focal)
      - LocationNearFocal                   : binary indicator (1 if dist_focal < dist_other)
      - RelMales                            : m_focal - m_other
      - TotalSize                           : n_focal + n_other
      - z_RelSize_log, z_LocationAdv, z_RelMales, z_TotalSize : standardized versions used in the model
      - dyad                                : kept for clustering of SEs

    The function drops rows with missing values in essential columns.
    """
    df = df.copy()

    # Required columns for analysis
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    missing_required = [c for c in required if c not in df.columns]
    if len(missing_required) > 0:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Drop rows with NA in required columns
    df = df.dropna(subset=required)

    # Ensure numeric types
    numeric_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows that turned NA after coercion
    df = df.dropna(subset=numeric_cols)

    # Create relative size: log ratio (use raw counts; counts here are >=5 so no zero issue)
    df['RelSize_log'] = np.log(df['n_focal'] / df['n_other'])

    # Location advantage: positive => contest closer to focal group's home center (dist_other > dist_focal)
    df['LocationAdv'] = df['dist_other'] - df['dist_focal']

    # Binary indicator: contest occurs nearer focal group's home center
    df['LocationNearFocal'] = (df['dist_focal'] < df['dist_other']).astype(int)

    # Relative number of adult males (could be important in contests)
    df['RelMales'] = df['m_focal'] - df['m_other']

    # Total size of participants (control)
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Standardize continuous predictors (z-scoring). Use population std (ddof=0) to be explicit.
    for col in ['RelSize_log', 'LocationAdv', 'RelMales', 'TotalSize']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, create zero column to avoid division by zero
            df['z_' + col] = 0.0
        else:
            df['z_' + col] = (df[col] - mean) / std

    # Ensure win is integer 0/1
    df['win'] = df['win'].astype(int)

    # Keep columns necessary for modeling and diagnostics
    keep_cols = ['win', 'RelSize_log', 'LocationAdv', 'LocationNearFocal', 'RelMales', 'TotalSize',
                 'z_RelSize_log', 'z_LocationAdv', 'z_RelMales', 'z_TotalSize', 'dyad',
                 'n_focal', 'n_other', 'm_focal', 'm_other', 'dist_focal', 'dist_other']
    # Some of these may already exist; return a dataframe with at least these columns (if any missing, they will be added above)
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting probability focal wins (win)
    from relative group size, location advantage, their interaction, and the control covariates.

    Uses cluster-robust standard errors clustered by 'dyad' to account for non-independence
    of contests within the same dyad.

    Model formula (Patsy-compatible):
      win ~ z_RelSize_log * z_LocationAdv + z_RelMales + z_TotalSize

    Returns the fitted model results with cluster-robust covariances.
    """
    # Check required columns
    required_model_cols = ['win', 'z_RelSize_log', 'z_LocationAdv', 'z_RelMales', 'z_TotalSize', 'dyad']
    missing = [c for c in required_model_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing columns required for modeling: {missing}")

    # Build model using formula interface
    formula = 'win ~ z_RelSize_log * z_LocationAdv + z_RelMales + z_TotalSize'

    # Fit binomial GLM
    model_glm = sm.GLM.from_formula(formula, data=df, family=sm.families.Binomial())
    res = model_glm.fit()

    # Obtain cluster-robust covariance (clustered by dyad)
    # .get_robustcov_results is used to adapt the covariance matrix
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # Fallback: if clustering fails, return the original fit but warn the user
        import warnings
        warnings.warn('Clustered SE computation failed; returning the original GLM results without clustering.')
        res_cluster = res

    # Optionally, attach predicted probabilities and marginal effects for interpretation
    df = df.copy()
    df['pred_prob'] = res_cluster.predict(df)

    # Return a dictionary with useful objects
    return {
        'results_clustered': res_cluster,
        'predicted_dataframe': df,
        'formula': formula
    }


