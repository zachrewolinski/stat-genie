from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/shuffle_names_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used for modeling.

    Expected original columns (from provided schema):
      - 'majority_first': outcome (1=unchosen option, 2=majority option, 3=minority option)
      - 'gender': 1=girl, 2=boy
      - 'culture': (mislabelled in schema) contains the child's age in years (4-14)
      - 'age': (mislabelled in schema) encodes whether the majority was demonstrated first (binary 0/1)
      - 'y': site id (1..8)

    Produces columns used in the model:
      - FollowedMajority: binary DV (1 if majority chosen)
      - Age: numeric age in years
      - Age_c: age centered at median
      - Age_c2: squared centered age (for quadratic term)
      - Site: categorical site id (strings 'Site_#')
      - Gender: binary control (0 = girl, 1 = boy)
      - MajorityDemonstratedFirst: control for demonstration order (as integer)
    """
    df = df.copy()

    # Drop rows missing any required raw inputs
    df = df.dropna(subset=['majority_first', 'culture', 'gender', 'age', 'y'])

    # Dependent variable: did the child follow the majority demonstration?
    # In the dataset majority_first == 2 indicates choosing the majority option
    df['FollowedMajority'] = (df['majority_first'] == 2).astype(int)

    # Age: the column named 'culture' in the schema contains child's age in years
    df['Age'] = pd.to_numeric(df['culture'], errors='coerce')

    # Demonstration order: the column named 'age' in the schema actually encodes
    # whether majority was demonstrated first (0/1). Keep as integer.
    # Ensure casting is safe by coerce -> fillna then int
    df['MajorityDemonstratedFirst'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)

    # Gender: map to binary 0/1 (0 = girl, 1 = boy)
    df['Gender'] = (df['gender'] == 2).astype(int)

    # Site: convert site id to a categorical string for use in formulas
    # Make sure y is numeric-ish before converting to int to avoid weird strings
    df['Site'] = 'Site_' + pd.to_numeric(df['y'], errors='coerce').fillna(0).astype(int).astype(str)

    # Restrict to plausible age range (4-14 years) in case of any odd values
    df = df[(df['Age'] >= 4) & (df['Age'] <= 14)]

    # Center age (use median to reduce influence of skew) and add quadratic term
    df['Age_c'] = df['Age'] - df['Age'].median()
    df['Age_c2'] = df['Age_c'] ** 2

    # Return only the columns needed for modeling (keeps small footprint)
    return df[[
        'FollowedMajority',
        'Age',
        'Age_c',
        'Age_c2',
        'Site',
        'Gender',
        'MajorityDemonstratedFirst'
    ]]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial (logistic) generalized linear model to test how following the majority
    changes with age and whether that developmental trajectory differs across sites.

    Model specification:
      FollowedMajority ~ Age_c * C(Site) + Age_c2 + Gender + MajorityDemonstratedFirst

    - Age_c * C(Site) tests whether age slopes differ by site (Age x Site interaction).
    - Age_c2 captures nonlinear (quadratic) age trends at the population level.
    - Gender and MajorityDemonstratedFirst are included as controls.

    We fit a GLM (binomial family) and also compute cluster-robust standard errors by Site.

    Returns a dict with the fitted glm and the cluster-robust result object.
    """
    # local imports for formula interface
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure Site is treated as categorical in the formula; C(Site) will handle it
    formula = 'FollowedMajority ~ Age_c * C(Site) + Age_c2 + Gender + MajorityDemonstratedFirst'

    # Fit the standard (MLE) GLM
    glm_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Cluster-robust standard errors grouped by Site to account for within-site correlation
    # Some versions of statsmodels do not provide get_robustcov_results on GLMResults,
    # so we obtain a second fit with cluster robust covariance using fit's cov_type/cov_kwds.
    cluster_groups = df['Site'].values
    glm_robust = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit(
        cov_type='cluster',
        cov_kwds={'groups': cluster_groups}
    )

    # Return both the original fit (for convenience) and the robust-cov results
    return {
        'glm_fit': glm_fit,
        'glm_robust': glm_robust
    }