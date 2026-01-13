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
    Transform the raw dataset into the analysis-ready dataframe containing the columns used in the model.

    Produces the following new/cleaned columns used in modeling:
      - MajorityChoice : binary DV (1 if y == 2 (majority), else 0)
      - age_c          : age centered around the sample mean
      - is_male        : 1 if gender == 2 (boy), 0 if gender == 1 (girl)
      - culture        : kept as-is (will be treated as categorical in the model)
      - majority_first : ensured numeric 0/1

    Drops rows with missing values in the essential variables (y, age, culture).
    """
    # Ensure we operate on a copy
    df = df.copy()

    # Keep only rows with the essential columns non-missing
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Create dependent variable: 1 if child chose the majority option (y == 2), else 0
    # According to the dataset: y: 1 = unchosen option, 2 = majority option, 3 = minority option
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Create a binary sex variable: is_male (1 if boy (gender==2), 0 if girl (gender==1))
    # If gender has other codings or missing, we coerce to numeric and set missing to NaN then drop above
    df['is_male'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is numeric 0/1
    # If majority_first is already 0/1 this is a no-op; coerce to integer where possible
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').fillna(0).astype(int)

    # Center age around the sample mean to improve interpretability of main effects and interactions
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()

    # Ensure culture is kept (will be treated as categorical in the model). Cast to integer category if possible
    # but keep the column name 'culture' (the model will use C(culture) in the formula)
    df['culture'] = df['culture'].astype(int)

    # Final check: drop any rows that became NA after coercion in columns used by the model
    df = df.dropna(subset=['MajorityChoice', 'age_c', 'is_male', 'majority_first', 'culture'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (GLM, binomial family) predicting the probability of choosing the majority option.

    The model tests whether age predicts majority choice, and whether that developmental effect differs across cultural sites
    by including an interaction between centered age and culture.

    Formula used:
      MajorityChoice ~ age_c * C(culture) + is_male + majority_first

    - age_c * C(culture) fits main effect of age and culture plus their interaction (age-by-culture), which addresses the
      research question about how reliance on majority preference develops over age across different cultural contexts.

    Returns the fitted results object (statsmodels GLMResults).
    """
    import statsmodels.formula.api as smf
    # Fit a logistic regression (binomial family)
    formula = 'MajorityChoice ~ age_c * C(culture) + is_male + majority_first'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = model.fit()

    # Return the fitted results object for inspection (summary, coefficients, confidence intervals, predicted probabilities, etc.)
    return results


