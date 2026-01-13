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
    Transform the raw dataset into a dataframe suitable for modeling.

    Produces the following columns required by the model:
      - IsMajority: binary outcome (1 if y == 2, else 0)
      - age_c: centered age (age - mean(age))
      - age_c2: squared centered age (to capture nonlinear age effects)
      - Culture: categorical culture/site identifier
      - GenderBoy: binary indicator 1 if boy (gender == 2), 0 if girl (gender == 1)
      - majority_first: ensures integer 0/1 for the demonstration order variable

    Drops rows with missing values in the key columns.
    """
    df = df.copy()

    # Drop rows missing the main variables needed for the analysis
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Dependent variable: did the child choose the majority option? (y == 2)
    df['IsMajority'] = (df['y'] == 2).astype(int)

    # Center age and add a quadratic term to allow nonlinear (e.g., accelerated) development
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c2'] = df['age_c'] ** 2

    # Gender: convert to a binary indicator where 1 = boy, 0 = girl
    # Original coding: 1 = girl, 2 = boy
    df['GenderBoy'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is 0/1 integer
    df['majority_first'] = df['majority_first'].astype(int)

    # Culture as categorical factor (keeps original IDs but as category for modeling)
    df['Culture'] = df['culture'].astype('category')

    # Return the transformed dataframe (keeps other columns too if needed for further checks)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) regression to predict choosing the majority option.

    Model specification:
      IsMajority ~ age_c + age_c2 + C(Culture) + age_c:C(Culture) + GenderBoy + majority_first

    - age_c and age_c2 model linear and quadratic age effects.
    - C(Culture) includes culture fixed effects to capture baseline differences across sites.
    - age_c:C(Culture) allows the age slope to vary by culture (i.e., tests whether developmental trajectories differ across cultural contexts).
    - GenderBoy and majority_first are included as controls.

    Returns the fitted results object with cluster-robust standard errors clustered by Culture (if available).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required_cols = ['IsMajority', 'age_c', 'age_c2', 'Culture', 'GenderBoy', 'majority_first']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # Formula with culture fixed effects and age-by-culture interactions
    formula = 'IsMajority ~ age_c + age_c2 + C(Culture) + age_c:C(Culture) + GenderBoy + majority_first'

    # Fit GLM (logistic regression)
    glm_res = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Attempt to produce cluster-robust SEs clustered by Culture
    # If clustering fails for any reason, fall back to the original glm results.
    try:
        clustered = glm_res.get_robustcov_results(cov_type='cluster', groups=df['Culture'])
    except Exception:
        clustered = glm_res

    # Return the results object (clustered if possible)
    return clustered


