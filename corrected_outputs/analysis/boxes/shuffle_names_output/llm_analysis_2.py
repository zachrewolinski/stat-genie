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
    Transform the raw dataset to a modeling dataframe that contains:
      - ChoseMajority: binary DV (1 if chose majority option, 0 otherwise)
      - Age: raw age in years (taken from column 'culture' in the provided schema)
      - Age_c: mean-centered age
      - Age_sq: squared centered age term for quadratic effect
      - Site: categorical site ID (from 'y')
      - Gender: recoded gender (0=girl, 1=boy) from 'gender'
      - MajorityDemoFirst: binary indicator (0/1) whether majority option was demonstrated first (from 'age' column in raw schema)

    Notes on the provided schema: the field names in the dataset appear misaligned with their descriptions in the metadata. Based on the metadata:
      - 'majority_first' is the outcome with codes 1=unchosen option, 2=majority option, 3=minority option
      - 'culture' contains the child's age in years
      - 'y' contains the site ID
      - 'age' is a binary indicator for whether majority was demonstrated first (0/1 in the raw data)
      - 'gender' is 1=girl, 2=boy

    The function will drop rows missing critical variables and create the derived columns used in the model.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['majority_first', 'culture', 'y', 'gender', 'age']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing critical values
    df = df.dropna(subset=['majority_first', 'culture', 'y'])

    # Dependent variable: did the child choose the majority-demonstrated option?
    # According to schema: 1 = unchosen option, 2 = majority option, 3 = minority option
    df['ChoseMajority'] = (df['majority_first'] == 2).astype(int)

    # Age: according to schema the 'culture' column stores age in years
    df['Age'] = pd.to_numeric(df['culture'], errors='coerce')

    # Drop rows with non-numeric or missing ages (after conversion)
    df = df.dropna(subset=['Age'])

    # Site (culture/site id): from column 'y' in schema
    # Convert to categorical (string) so fixed-effects in formulas treat it as factor
    df['Site'] = df['y'].astype('category')

    # Gender: original coding 1=girl, 2=boy. Recode to 0=girl, 1=boy
    df['Gender'] = df['gender'].map({1: 0, 2: 1})
    # If there are other codes, coerce to NaN and drop
    df = df[df['Gender'].isin([0, 1])]

    # Majority demonstration order: original 'age' column in raw schema is a binary indicator
    # We keep it as MajorityDemoFirst (0/1). If it's not 0/1, coerce to binary by treating nonzero as 1
    df['MajorityDemoFirst'] = pd.to_numeric(df['age'], errors='coerce')
    df['MajorityDemoFirst'] = df['MajorityDemoFirst'].fillna(0).astype(int)
    df.loc[~df['MajorityDemoFirst'].isin([0, 1]), 'MajorityDemoFirst'] = df.loc[~df['MajorityDemoFirst'].isin([0, 1]), 'MajorityDemoFirst'].apply(lambda x: 1 if x != 0 else 0)

    # Center age and create quadratic term for nonlinear effects
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['Age_sq'] = df['Age_c'] ** 2

    # Keep only columns needed for modeling (this helps downstream code know column names exactly)
    df = df[['ChoseMajority', 'Age', 'Age_c', 'Age_sq', 'Site', 'Gender', 'MajorityDemoFirst']]

    # Reset index (optional) and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a generalized linear model (logistic regression) predicting the binary outcome ChoseMajority.

    Primary predictors:
      - Age_c (centered age) and Age_sq (quadratic term) to capture nonlinear development
      - Gender and MajorityDemoFirst as covariates
      - Site as a categorical fixed effect to control for cultural context
      - Age_c x Site interactions to test whether age-related change differs across cultures

    The model is a binomial GLM (logistic regression) fit via statsmodels.
    Returns the fitted model object (statsmodels.GLMResults).
    """
    import statsmodels.formula.api as smf

    # Check that required columns exist
    required = ['ChoseMajority', 'Age_c', 'Age_sq', 'Gender', 'MajorityDemoFirst', 'Site']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in transformed dataframe: {missing}")

    # Formulate model: include Site as categorical fixed effect and Age_c x Site interactions
    # Note: Age_sq is included as a main effect (no interactions) to allow curvature in age trajectories.
    formula = 'ChoseMajority ~ Age_c + Age_sq + Gender + MajorityDemoFirst + C(Site) + Age_c:C(Site)'

    # Fit binomial GLM (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model object; user can call model.summary() or inspect params, pvalues, etc.
    return model


