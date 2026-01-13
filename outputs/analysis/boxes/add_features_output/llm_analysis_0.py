from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe. Returns a dataframe that contains all columns
    referenced in the conceptual variables and modeling code.

    Output columns created/kept:
      - MajorityChoice : binary DV (1 if y==2 (majority), else 0)
      - age_c, age_c_sq : mean-centered age and squared term
      - culture_cat : culture as categorical variable (kept as category dtype)
      - gender_female : female=1, male=0
      - majority_first : indicator (0/1) whether majority demonstrated first
      - religiousness_z, calworks_z : z-scored covariates
      - school : school identifier (kept for clustering)
    """
    df = df.copy()

    # Keep only rows with the essentials for the primary analysis
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Dependent variable: did the child choose the majority option?
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Age: center and include quadratic term for nonlinear trajectories
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c_sq'] = df['age_c'] ** 2

    # Culture: keep as categorical variable (site-level moderator)
    df['culture_cat'] = df['culture'].astype('category')

    # Gender: code female = 1, male = 0 (dataset codes: 1=girl, 2=boy)
    df['gender_female'] = (df['gender'] == 1).astype(int)

    # majority_first should be 0/1; fill NA with 0 (if appropriate) then cast
    if 'majority_first' in df.columns:
        df['majority_first'] = df['majority_first'].fillna(0).astype(int)
    else:
        df['majority_first'] = 0

    # Z-score continuous covariates (religiousness and calworks) to aid interpretation
    if 'religiousness' in df.columns:
        # avoid division by zero if constant
        denom = df['religiousness'].std(ddof=0)
        denom = denom if denom != 0 else 1.0
        df['religiousness_z'] = (df['religiousness'] - df['religiousness'].mean()) / denom
    else:
        df['religiousness_z'] = np.nan

    if 'calworks' in df.columns:
        denom = df['calworks'].std(ddof=0)
        denom = denom if denom != 0 else 1.0
        df['calworks_z'] = (df['calworks'] - df['calworks'].mean()) / denom
    else:
        df['calworks_z'] = np.nan

    # Ensure school column exists for clustering; if missing, create a placeholder
    if 'school' not in df.columns:
        df['school'] = 'unknown_school'

    # Drop rows with missing values in the predictors/controls we will use in the model
    required = ['MajorityChoice', 'age_c', 'age_c_sq', 'culture_cat', 'gender_female', 'majority_first', 'religiousness_z', 'calworks_z', 'school']
    df = df.dropna(subset=required)

    # Return only the columns required for modeling (plus any original columns you want to retain)
    keep_cols = required
    return df[keep_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression model predicting choice of the majority option.

    Model specification (primary):
      MajorityChoice ~ age_c + age_c_sq + C(culture_cat) + age_c:C(culture_cat)
                        + gender_female + majority_first + religiousness_z + calworks_z

    We include age × culture interactions to estimate culture-specific developmental trajectories.
    We fit a binomial GLM and compute cluster-robust standard errors clustered by school.

    Returns:
      - A statsmodels results object with cluster-robust covariance if available.
    """
    import statsmodels.formula.api as smf

    # formula: main effects + culture fixed effects + age x culture interactions + controls
    formula = (
        'MajorityChoice ~ age_c + age_c_sq + C(culture_cat) + age_c:C(culture_cat) '
        '+ gender_female + majority_first + religiousness_z + calworks_z'
    )

    # Fit binomial GLM
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute cluster-robust SEs by school (if possible). If clustering fails, fall back to original results.
    try:
        results = glm_binom.get_robustcov_results(cov_type='cluster', groups=df['school'])
    except Exception:
        # Some environments / statsmodels versions may not support get_robustcov_results on GLMResults
        # In that case return the original GLM fit object.
        results = glm_binom

    return results


