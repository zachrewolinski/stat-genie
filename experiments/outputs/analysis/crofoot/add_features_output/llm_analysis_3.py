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
    Transform the raw capuchin contest dataframe to the variables used in modeling.

    Produces these new columns used in the model:
      - RelSize_z: z-scored n_focal / n_other (continuous independent variable)
      - ContestLocation: categorical ('FocalHome', 'OtherHome', 'Neutral') (independent variable)
      - MaleDiff_z: z-scored (m_focal - m_other) (control)
      - FemaleDiff_z: z-scored (f_focal - f_other) (control)
      - DistDiff_z: z-scored (dist_other - dist_focal) (control)
      - dyad: categorical dyad id (control)

    Keeps the original 'win' column as the dependent variable.
    """
    df = df.copy()

    # Required columns for the analysis
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                     'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']

    # Drop rows with missing values in critical fields
    df = df.dropna(subset=required_cols)

    # Compute relative size measures
    # Use ratio (continuous) and also keep difference for diagnostics if needed
    df['RelSizeRatio'] = df['n_focal'] / df['n_other']
    df['RelSizeDiff'] = df['n_focal'] - df['n_other']

    # Compute sex composition differences
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Compute distance difference such that positive => focal is closer to its home center than other
    df['DistDiff'] = df['dist_other'] - df['dist_focal']

    # Define ContestLocation based on which group is nearer its home-range center.
    # If the distances are very similar (within threshold), mark as 'Neutral'.
    # Threshold chosen as 50 meters (reasonable given distance scale in the dataset).
    thresh = 50
    df['ContestLocation'] = np.where(
        np.abs(df['dist_focal'] - df['dist_other']) <= thresh,
        'Neutral',
        np.where(df['dist_focal'] < df['dist_other'], 'FocalHome', 'OtherHome')
    )

    # Convert dyad to categorical
    df['dyad'] = df['dyad'].astype('category')

    # Z-score continuous predictors (use population std ddof=0 to match typical standardization for modeling)
    def zscore(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)

    df['RelSize_z'] = zscore(df['RelSizeRatio'])
    df['MaleDiff_z'] = zscore(df['MaleDiff'])
    df['FemaleDiff_z'] = zscore(df['FemaleDiff'])
    df['DistDiff_z'] = zscore(df['DistDiff'])

    # Ensure ContestLocation is categorical with a defined order (optional)
    df['ContestLocation'] = pd.Categorical(df['ContestLocation'], categories=['FocalHome', 'Neutral', 'OtherHome'])

    # Final returned dataframe includes all new columns plus the dependent variable and dyad
    # (we keep original columns as well in case of downstream checks)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (GLM, binomial) predicting focal group win.

    Model specification:
      win ~ RelSize_z * C(ContestLocation) + MaleDiff_z + FemaleDiff_z + DistDiff_z + C(dyad)

    - The interaction term tests whether the effect of relative group size on winning
      depends on contest location (home/other/neutral).
    - dyad is included as a categorical fixed effect to absorb dyad-specific unobserved heterogeneity.
    - Cluster-robust standard errors by dyad are returned to account for non-independence of contests within dyads.

    Returns the fitted results object with cluster-robust covariance.
    """
    import statsmodels.formula.api as smf

    # Make sure the key predictors exist
    required_model_cols = ['win', 'RelSize_z', 'ContestLocation', 'MaleDiff_z', 'FemaleDiff_z', 'DistDiff_z', 'dyad']
    missing = [c for c in required_model_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for model: {missing}")

    # Formula with interaction between relative size and location
    formula = 'win ~ RelSize_z * C(ContestLocation) + MaleDiff_z + FemaleDiff_z + DistDiff_z + C(dyad)'

    # Fit GLM (logistic)
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = model_glm.fit()

    # Compute cluster-robust covariance by dyad and return adjusted results
    # statsmodels allows obtaining robust cov results from the fitted model
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If clustering fails for some reason, return the original result but warn the user
        print("Warning: could not compute cluster-robust SEs by dyad; returning standard GLM fit.")
        res_clust = res

    # Print a short summary for quick inspection (can be removed depending on use-case)
    print(res_clust.summary())

    return res_clust


