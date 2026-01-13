from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Keep only rows with required variables
    required_cols = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=required_cols)

    # Dependent variable: majority chosen (y == 2)
    # y: 1 = unchosen option, 2 = majority option, 3 = minority option
    df['MajorityChosen'] = (df['y'] == 2).astype(int)

    # Age: center and add quadratic term for potential nonlinearity
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_sq'] = df['age_c'] ** 2

    # Culture/site as categorical (keep original numeric IDs but cast to category dtype)
    df['culture'] = df['culture'].astype('category')

    # Gender: original coding 1 = girl, 2 = boy. Create binary is_boy (1 = boy, 0 = girl)
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # majority_first should be numeric 0/1; coerce to int
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').fillna(0).astype(int)

    # Final subset to ensure no NA in key model columns
    model_cols = ['MajorityChosen', 'age_c', 'age_sq', 'culture', 'is_boy', 'majority_first']
    df = df.dropna(subset=model_cols)

    # Return dataframe with the new columns available for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting choice of the majority option.
    We model a main effect of age (linear + quadratic), culture (categorical), and their interaction
    (age_c * culture) to test whether developmental trajectories differ across cultural contexts.
    Controls: is_boy, majority_first.

    Returns the fitted GLMResults object (Binomial family).
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure culture is categorical (patsy will treat C(culture) as categorical)
    df = df.copy()
    df['culture'] = df['culture'].astype('category')

    # Build formula: include linear and quadratic age terms and interactions of linear age with culture.
    # We include interactions for the linear age term with culture to test different slopes across cultures.
    # Quadratic term is included without culture interactions by default (can be extended if desired).
    formula = 'MajorityChosen ~ age_c + age_sq + C(culture) + age_c:C(culture) + is_boy + majority_first'

    # Fit GLM with binomial family (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model results so the caller can inspect summary, params, pvalues, etc.
    return model


