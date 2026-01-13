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
    Transform the raw dataset into the analysis dataframe. The returned dataframe contains the
    columns used in the model: MajorityChoice (binary DV), age_c (centered age), age_sq (quadratic),
    IsGirl (gender binary), majority_first (order), and culture (categorical moderator).
    """
    # Keep a copy of the input DataFrame
    df = df.copy()

    # Drop rows missing the core variables needed for analysis
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Create binary dependent variable: 1 if child chose the majority option (y == 2), else 0
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Center age to improve interpretability and reduce collinearity with age_sq
    df['age_c'] = df['age'] - df['age'].mean()
    # Quadratic term to capture non-linear developmental change
    df['age_sq'] = df['age_c'] ** 2

    # Gender: create binary indicator for girl (1 if girl, 0 if boy or other)
    # According to the schema: 1 = girl, 2 = boy
    df['IsGirl'] = (df['gender'] == 1).astype(int)

    # Ensure majority_first is binary integer (if missing, keep as NA and downstream model will drop)
    # If majority_first is not present for some rows, this will raise; the dataset schema includes it.
    df['majority_first'] = df['majority_first'].astype(int)

    # Ensure culture is categorical (keeps original numeric codes but marks as category for modeling)
    df['culture'] = df['culture'].astype('category')

    # Drop any rows that still have NA in columns we will use in the model
    df = df.dropna(subset=['MajorityChoice', 'age_c', 'age_sq', 'IsGirl', 'majority_first', 'culture'])

    # Return the transformed dataframe ready for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) regression predicting the probability of choosing the majority option.
    The model includes: main effect of age (centered), quadratic age (age_sq), interaction between age and culture
    to test whether developmental slopes differ across cultures, and controls for gender and order (majority_first).

    Returns a fitted model object with cluster-robust standard errors by culture.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Formula: allow age slope to vary by culture via interaction; include quadratic age as a global effect;
    # control for gender and demonstration order. C(culture) treats culture as categorical.
    formula = 'MajorityChoice ~ age_c * C(culture) + age_sq + IsGirl + majority_first'

    # Fit a binomial GLM (logistic regression)
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute cluster-robust covariance (clustered on culture) to account for within-site dependence
    try:
        results_cluster = glm_binom.get_robustcov_results(cov_type='cluster', groups=df['culture'])
    except Exception:
        # If clustered covariance fails for any reason, return the original fitted model
        results_cluster = glm_binom

    # Return the model results object (with robust clustered SEs when available)
    return results_cluster


