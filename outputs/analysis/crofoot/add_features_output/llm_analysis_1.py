from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin intergroup contest dataframe into a modeling-ready dataframe.

    Produces the following new columns used by the model:
      - RelativeSize = n_focal - n_other
      - LocAdv = dist_other - dist_focal  (positive = focal closer to its home center)
      - TotalSize = n_focal + n_other
      - RelMales = m_focal - m_other
      - RelFemales = f_focal - f_other
      - z_... standardized versions of the continuous predictors
      - dyad as a categorical variable

    Drops rows with missing values in key columns required for the analysis.
    """
    df = df.copy()

    # Required columns for the analysis
    required_cols = [
        'win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]

    # Drop rows missing any of the required values
    df = df.dropna(subset=required_cols)

    # Compute raw derived variables
    df['RelativeSize'] = df['n_focal'] - df['n_other']
    df['LocAdv'] = df['dist_other'] - df['dist_focal']
    df['TotalSize'] = df['n_focal'] + df['n_other']
    df['RelMales'] = df['m_focal'] - df['m_other']
    df['RelFemales'] = df['f_focal'] - df['f_other']

    # Standardize (z-score) the continuous predictors for interpretability and numeric stability
    # Use population std (ddof=0) to avoid potential small-sample complications; ddof=1 is also acceptable.
    def zscore(s: pd.Series) -> pd.Series:
        if s.std(ddof=0) == 0 or np.isclose(s.std(ddof=0), 0):
            return (s - s.mean())  # if zero variance, return mean-centered (all zeros)
        return (s - s.mean()) / s.std(ddof=0)

    df['z_RelativeSize'] = zscore(df['RelativeSize'])
    df['z_LocAdv'] = zscore(df['LocAdv'])
    df['z_TotalSize'] = zscore(df['TotalSize'])
    df['z_RelMales'] = zscore(df['RelMales'])
    df['z_RelFemales'] = zscore(df['RelFemales'])

    # Ensure dyad is a categorical variable for fixed effects
    df['dyad'] = df['dyad'].astype('category')

    # Keep only the columns necessary for modeling (plus commonly useful originals)
    keep_cols = [
        'win',
        'z_RelativeSize',
        'z_LocAdv',
        'z_TotalSize',
        'z_RelMales',
        'z_RelFemales',
        'dyad',
        # raw versions kept for diagnostics if needed
        'RelativeSize', 'LocAdv', 'TotalSize', 'RelMales', 'RelFemales'
    ]

    # Some datasets may include extra rows; select only those columns that exist after transformation
    keep_cols_existing = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols_existing]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic (binomial) generalized linear model predicting focal-group win.

    Primary specification estimate:
      win ~ z_RelativeSize * z_LocAdv + z_RelMales + z_RelFemales + z_TotalSize + C(dyad)

    The interaction term tests whether the effect of relative group size on win-probability
    depends on contest location (home advantage).

    Returns the fitted GLMResults object (statsmodels) and prints the summary.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns are present
    req = ['win', 'z_RelativeSize', 'z_LocAdv', 'z_RelMales', 'z_RelFemales', 'z_TotalSize', 'dyad']
    missing = [c for c in req if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modelling: {missing}")

    # Model formula: binomial GLM with dyad fixed effects
    formula = ('win ~ z_RelativeSize * z_LocAdv '
               '+ z_RelMales + z_RelFemales + z_TotalSize + C(dyad)')

    # Fit binomial GLM (logit link by default in statsmodels for GLM family=Binomial)
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Print summary
    print(glm_binom.summary())

    # For inference robust to dyad clustering, compute cluster-robust covariance if desired
    # (Requires at least a few clusters; with very few clusters results may be unstable.)
    try:
        clustered = glm_binom.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
        print('\nCluster-robust (by dyad) results:')
        print(clustered.summary())
        # Return both the original fit and the clustered-covariance results object
        return {'fit': glm_binom, 'cluster_robust': clustered}
    except Exception:
        # If clustering fails (e.g., too few clusters), just return the regular fit
        return {'fit': glm_binom}


