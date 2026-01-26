from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/anonymize_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Steps performed:
    - Rename columns to meaningful names used in the model.
    - Drop rows missing any variables required for modeling.
    - Compute derived variables: SizeRatio, SizeDiff, MaleDiff, FemaleDiff, LocationAdv.
    - Create standardized (z-scored) versions of continuous predictors used in the model.
    - Create the interaction term used in the model (SizeRatio_z * LocationAdv_z).

    Returns the transformed dataframe containing all columns referenced in the conceptual variables.
    """
    df = df.copy()

    # Rename columns to descriptive names used in modeling
    df = df.rename(columns={
        'feature1': 'focal_id',
        'feature2': 'other_id',
        'feature3': 'dyad_id',
        'feature4': 'focal_win',
        'feature5': 'focal_dist',
        'feature6': 'other_dist',
        'feature7': 'focal_n',
        'feature8': 'other_n',
        'feature9': 'focal_males',
        'feature10': 'other_males',
        'feature11': 'focal_females',
        'feature12': 'other_females'
    })

    # Required columns for modeling
    required_cols = [
        'focal_win', 'focal_dist', 'other_dist', 'focal_n', 'other_n',
        'focal_males', 'other_males', 'focal_females', 'other_females', 'dyad_id'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Ensure correct dtypes
    # focal_win should be binary (0/1)
    df['focal_win'] = df['focal_win'].astype(int)
    df['dyad_id'] = df['dyad_id'].astype('category')

    # Derived predictors
    # Relative size: ratio and difference
    df['SizeRatio'] = df['focal_n'] / df['other_n']
    df['SizeDiff'] = df['focal_n'] - df['other_n']
    # Sex composition differences
    df['MaleDiff'] = df['focal_males'] - df['other_males']
    df['FemaleDiff'] = df['focal_females'] - df['other_females']
    # Location advantage: positive values indicate contest is closer to focal group's center
    df['LocationAdv'] = df['other_dist'] - df['focal_dist']
    # Simple binary indicator if focal group is closer
    df['focal_closer'] = (df['focal_dist'] < df['other_dist']).astype(int)

    # Standardize (z-score) continuous predictors that we will include in the model
    z_cols = ['SizeRatio', 'SizeDiff', 'MaleDiff', 'FemaleDiff', 'LocationAdv']
    for col in z_cols:
        # Use population std (ddof=0) to be explicit and stable for small samples
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            # avoid division by zero; set z to 0 if no variance
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Interaction between standardized size ratio and standardized location advantage
    df['SizeRatio_z_x_LocationAdv_z'] = df['SizeRatio_z'] * df['LocationAdv_z']

    # Keep columns necessary for modeling and diagnostics
    keep_cols = [
        'focal_id', 'other_id', 'dyad_id', 'focal_win',
        'focal_dist', 'other_dist', 'focal_n', 'other_n',
        'focal_males', 'other_males', 'focal_females', 'other_females',
        'SizeRatio', 'SizeDiff', 'MaleDiff', 'FemaleDiff', 'LocationAdv',
        'SizeRatio_z', 'SizeDiff_z', 'MaleDiff_z', 'FemaleDiff_z', 'LocationAdv_z',
        'SizeRatio_z_x_LocationAdv_z', 'focal_closer'
    ]

    # Subset to keep only those columns (drop any extras)
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (binomial GLM with logit link) predicting probability that the focal group won.

    Model specification (primary):
      focal_win ~ SizeRatio_z * LocationAdv_z + MaleDiff_z + FemaleDiff_z + C(dyad_id)

    - The formula includes the interaction between relative group size and location advantage.
    - Dyad identity is included as a categorical fixed effect to account for pair-specific heterogeneity.
    - After fitting the GLM, clustered (robust) standard errors clustered by dyad_id are computed.

    Returns a dictionary containing the fitted model, clustered-robust results, a coefficient table, and a textual summary.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Work on a copy
    df = df.copy()

    # Ensure dyad_id is categorical for formula interface
    df['dyad_id'] = df['dyad_id'].astype('category')

    # Formula with interaction and dyad fixed effects
    formula = 'focal_win ~ SizeRatio_z * LocationAdv_z + MaleDiff_z + FemaleDiff_z + C(dyad_id)'

    # Fit binomial GLM (logit link)
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Obtain cluster-robust covariance (clustered by dyad_id)
    # Use get_robustcov_results to compute robust covariances
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['dyad_id'])
    except Exception:
        # Fallback: if clustering fails, return the standard GLM results as 'res_cluster'
        res_cluster = res

    # Prepare a coefficients table (pandas DataFrame) from the clustered results
    try:
        coef_table = res_cluster.summary2().tables[1]
    except Exception:
        # If summary2 is not available for the robust object, create a manual table
        coef = res_cluster.params
        se = res_cluster.bse
        z = coef / se
        p = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
        coef_table = pd.DataFrame({
            'coef': coef,
            'std_err': se,
            'z': z,
            'P>|z|': p
        })

    # Return useful objects; in interactive use these let the analyst inspect both raw and robust results
    return {
        'model_fitted': res,
        'model_cluster_robust': res_cluster,
        'coef_table': coef_table,
        'summary_text': res_cluster.summary().as_text()
    }


