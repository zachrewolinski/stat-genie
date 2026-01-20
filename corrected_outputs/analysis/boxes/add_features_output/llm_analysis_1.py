from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep only rows with required fields
    req_cols = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=req_cols)

    # Dependent variable: binary indicator for choosing the majority option
    # In the schema y: 1 = unchosen option, 2 = majority option, 3 = minority option
    df['chosen_majority'] = (df['y'] == 2).astype(int)

    # Age: center for numerical stability and create quadratic term to capture nonlinear development
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    mean_age = df['age'].mean()
    df['age_c'] = df['age'] - mean_age
    df['age_c2'] = df['age_c'] ** 2

    # Culture: ensure consistent type (keep numeric ids but allow categorical modeling)
    # If culture is not integer, coerce to integer codes
    df['culture'] = pd.to_numeric(df['culture'], errors='coerce').astype(int)

    # Gender: keep as is but ensure no missing and treat as categorical in model
    # Map to strings for clarity (optional for modeling with C(gender))
    # Original coding: 1=girl, 2=boy
    df['gender'] = df['gender'].map({1: 'girl', 2: 'boy'}).astype('category')

    # majority_first: ensure binary numeric (0/1)
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').fillna(0).astype(int)

    # Final drop of any rows that became NA after transforms
    model_cols = ['chosen_majority', 'age_c', 'age_c2', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=model_cols)

    # Return the dataframe that contains exactly the columns referenced in the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Formula: allow culture-specific linear and quadratic age effects via interaction
    # Controls: gender and whether majority was shown first
    formula = 'chosen_majority ~ C(culture) * (age_c + age_c2) + C(gender) + majority_first'

    # Fit a binomial GLM (logistic regression)
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Obtain cluster-robust standard errors clustered by culture (accounts for within-culture dependence)
    try:
        robust_results = glm_model.get_robustcov_results(cov_type='cluster', groups=df['culture'])
    except Exception:
        # Fallback to default results if clustering fails
        robust_results = glm_model

    # Return the fitted model with robust SEs (or the plain fit if robust cov failed)
    return robust_results


