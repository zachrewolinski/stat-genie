from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/shuffle_names_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw dataset into a dataframe with the columns needed for modeling.

    Expected input columns (from provided schema):
      - 'majority_first' : outcome code (1=unchosen option, 2=majority option, 3=minority option)
      - 'culture'        : (per schema) actually contains the child's age in years
      - 'age'            : binary indicator whether majority option was demonstrated first (0/1)
      - 'gender'         : 1=girl, 2=boy
      - 'y'              : site id (integer)

    Produced columns:
      - 'ChoseMajority'      : binary DV (1 = chose majority, 0 = otherwise)
      - 'Age'                : numeric age in years
      - 'Age_c'              : age centered around the sample mean
      - 'Site'               : categorical site id (string)
      - 'Gender'             : recoded gender (1 = girl, 0 = boy)
      - 'MajorityShownFirst' : 0/1 order-control variable
    """
    df = df.copy()

    # Drop rows with missing values in columns required for this analysis
    df = df.dropna(subset=['majority_first', 'culture', 'gender', 'y', 'age'])

    # Dependent variable: did the child choose the majority option?
    # Per schema: majority_first == 2 indicates choosing the majority option
    df['ChoseMajority'] = (df['majority_first'] == 2).astype(int)

    # Independent variable: Age (the 'culture' column in the provided schema holds age in years)
    df['Age'] = pd.to_numeric(df['culture'], errors='coerce')

    # Center age to aid interpretation and numerical stability in interaction models
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Site / cultural context: use 'y' as the site identifier and coerce to a categorical string
    df['Site'] = df['y'].astype(int).astype(str)

    # Gender: map 1=girl -> 1, 2=boy -> 0
    df['Gender'] = df['gender'].map({1: 1, 2: 0}).astype(float)

    # MajorityShownFirst: the provided 'age' column indicates whether majority was shown first per schema
    # Ensure it's 0/1 integer
    df['MajorityShownFirst'] = df['age'].astype(int)

    # Keep only the columns required for the statistical model
    df = df[[
        'ChoseMajority',
        'Age',
        'Age_c',
        'Site',
        'Gender',
        'MajorityShownFirst'
    ]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fits a logistic regression model predicting the probability of choosing the majority option.

    Model specification (fixed-effects approach):
      ChoseMajority ~ Age_c * C(Site) + Gender + MajorityShownFirst

    This specification estimates site-specific age slopes (via Age_c:C(Site) interaction) to
    test whether the developmental trajectory (change in majority reliance with age) differs
    across cultural contexts. Gender and demonstration-order (MajorityShownFirst) are included
    as covariates.

    Returns the fitted GLM (binomial family) result object from statsmodels.
    """
    import statsmodels.formula.api as smf

    # Ensure C(Site) is treated as categorical in the formula; Age_c interacts with Site
    formula = 'ChoseMajority ~ Age_c * C(Site) + Gender + MajorityShownFirst'

    # Fit a binomial GLM (logistic regression)
    model_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model object (has .summary(), .params, .predict(), etc.)
    return model_fit


