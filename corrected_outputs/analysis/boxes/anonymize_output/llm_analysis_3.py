from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/anonymize_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.

    Produces columns:
      - MajorityChoice: int (1 if feature1 == 2 (majority), else 0)
      - Age: numeric (feature3)
      - Age_c: Age centered around the sample mean
      - Gender_male: int (1 if feature2 == 2 (boy), 0 if feature2 == 1 (girl))
      - MajorityFirst: int copy of feature4 (0/1)
      - Site: categorical label for site (e.g., 'site_1')

    Drops rows missing any of the variables required for the model.
    """
    # Ensure a copy
    df = df.copy()

    # Rename original features for clarity (optional) and ensure expected columns exist
    # feature1: outcome (1=unchosen, 2=majority, 3=minority)
    # feature2: gender (1=girl, 2=boy)
    # feature3: age in years
    # feature4: whether majority was demonstrated first (0/1)
    # feature5: site id (1..8)

    # Drop rows with missing essential variables
    df = df.dropna(subset=['feature1', 'feature3', 'feature5'])

    # Create MajorityChoice binary outcome: 1 if child chose the majority (feature1 == 2), else 0
    df['MajorityChoice'] = (df['feature1'] == 2).astype(int)

    # Age: copy and ensure numeric
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')

    # Drop rows where Age conversion failed
    df = df.dropna(subset=['Age'])

    # Center age (use mean-centering to aid interpretation)
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Gender_male: 1 if boy (feature2 == 2), 0 if girl (feature2 == 1). If missing, set to NaN and drop later.
    df['Gender_male'] = df['feature2'].map({1: 0, 2: 1}).astype('Int64')

    # MajorityFirst: ensure integer 0/1 from feature4
    df['MajorityFirst'] = pd.to_numeric(df['feature4'], errors='coerce').fillna(0).astype(int)

    # Site: create categorical labels (strings) to use in formula C(Site)
    df['Site'] = 'site_' + df['feature5'].astype(int).astype(str)

    # Drop rows with missing Gender if present (we keep only complete cases for controls)
    df = df.dropna(subset=['Gender_male'])
    # Convert Gender_male to int (from pandas Int64)
    df['Gender_male'] = df['Gender_male'].astype(int)

    # Keep only the columns needed for modeling (makes downstream code clearer)
    df = df[['MajorityChoice', 'Age', 'Age_c', 'Gender_male', 'MajorityFirst', 'Site']]

    # Optionally reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting MajorityChoice.

    Formula:
      MajorityChoice ~ Age_c * C(Site) + Gender_male + MajorityFirst

    This estimates a separate age slope for each site (culture) via the Age_c:C(Site) interaction,
    controlling for gender and demonstration order. Cluster-robust standard errors by Site are
    computed to account for within-site correlation.

    Returns the fitted results object with cluster-robust covariances applied.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure Site is categorical
    df = df.copy()
    df['Site'] = df['Site'].astype('category')

    # Define formula: allow Age_c slope to vary by site via interaction with categorical Site
    formula = 'MajorityChoice ~ Age_c * C(Site) + Gender_male + MajorityFirst'

    # Fit binomial GLM (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    res = model.fit()

    # Obtain cluster-robust covariance results clustered by Site category
    # statsmodels requires groups to be an array; use the numeric codes of the categorical Site
    groups = df['Site'].cat.codes if hasattr(df['Site'], 'cat') else pd.Categorical(df['Site']).codes

    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=groups)
    except Exception:
        # If clustering fails for any reason, fall back to the original results
        res_cluster = res

    # Print a brief summary for convenience (caller can also inspect returned object)
    print(res_cluster.summary())

    return res_cluster


