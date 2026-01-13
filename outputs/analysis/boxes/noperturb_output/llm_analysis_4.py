from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/noperturb_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.
    Produces the following columns required for modeling:
      - majority_choice : binary DV (1 if y==2 else 0)
      - age_centered    : centered age (continuous IV)
      - culture         : categorical site ID (used as moderator)
      - gender_f        : gender coded 0 (girl) / 1 (boy)
      - majority_first  : binary indicator (0/1) for demonstration order

    Rows with missing values in the required fields are dropped.
    """
    # Work on a copy
    df = df.copy()

    # Required columns: 'y', 'age', 'culture', 'gender', 'majority_first'
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=required)

    # Dependent variable: majority_choice (1 if chose majority option coded as y==2)
    df['majority_choice'] = (df['y'] == 2).astype(int)

    # Independent variable: age (centered)
    # Keep original age column but create a mean-centered version for modeling
    df['age_centered'] = df['age'] - df['age'].mean()

    # Culture: ensure categorical type for modeling (C(culture) will be used)
    # Keep numeric codes but convert dtype to 'category' to make intentions explicit
    df['culture'] = df['culture'].astype('category')

    # Gender: map to 0/1 with clear naming (0 = girl, 1 = boy)
    # Original coding: 1 = girl, 2 = boy
    df['gender_f'] = df['gender'].map({1: 0, 2: 1})
    # If any unexpected gender codes remain, drop them
    df = df.dropna(subset=['gender_f'])
    df['gender_f'] = df['gender_f'].astype(int)

    # majority_first: ensure binary integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Final check: keep only rows with finite values in columns used for model
    model_cols = ['majority_choice', 'age_centered', 'culture', 'gender_f', 'majority_first']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial logistic regression predicting majority_choice from age,
    with culture as a moderator (age x culture interaction). Controls: gender_f, majority_first.

    The model formula is:
      majority_choice ~ age_centered * C(culture) + gender_f + majority_first

    We fit a logistic regression using statsmodels' formula interface and then
    compute cluster-robust standard errors clustered by culture.

    Returns the statsmodels results object with cluster-robust covariances applied.
    """
    import statsmodels.formula.api as smf

    # Ensure culture is available for clustering as group labels (use the original category codes)
    # If culture is categorical dtype, convert groups to its codes for clustering
    if hasattr(df['culture'].dtype, 'categories'):
        cluster_groups = df['culture'].cat.codes
    else:
        cluster_groups = df['culture']

    # Define formula including interaction
    formula = 'majority_choice ~ age_centered * C(culture) + gender_f + majority_first'

    # Fit logistic regression (maximum likelihood)
    model_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    # Convert to cluster-robust covariance (cluster by culture)
    try:
        results_clustered = model_fit.get_robustcov_results(cov_type='cluster', groups=cluster_groups)
    except Exception:
        # Fallback: return the original model_fit if robust cov can't be computed
        results_clustered = model_fit

    return results_clustered


