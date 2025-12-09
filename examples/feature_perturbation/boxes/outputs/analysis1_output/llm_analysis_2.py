from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/boxes/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.

    Produces the following columns required by the model:
      - ChoseMajority: binary outcome (1 if y==2, else 0)
      - Age_c: age centered around the sample mean
      - Age_c2: squared centered age (to capture nonlinearity)
      - IsMale: binary male indicator (1=male, 0=female)
      - MajorityFirst: binary indicator copied from majority_first
      - Culture: categorical site identifier (kept as original integer/category)

    Drops rows with missing values in any of the relevant columns.
    """

    # Keep a copy to avoid modifying original
    df = df.copy()

    # Required columns in raw data: 'y', 'age', 'gender', 'majority_first', 'culture'
    required = ['y', 'age', 'gender', 'majority_first', 'culture']
    # Drop rows with missing required variables
    df = df.dropna(subset=required)

    # Dependent variable: did the child choose the majority option? (y==2)
    df['ChoseMajority'] = (df['y'] == 2).astype(int)

    # Independent variable: center age to aid interpretation
    df['Age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()
    # Add a quadratic term to allow non-linear development across age
    df['Age_c2'] = df['Age_c'] ** 2

    # Controls
    # gender: original coding 1=girl, 2=boy -> create IsMale (1=boy, 0=girl)
    df['IsMale'] = (df['gender'] == 2).astype(int)

    # majority_first already 0/1 in dataset; ensure integer type
    df['MajorityFirst'] = df['majority_first'].astype(int)

    # Culture as categorical (keep original numeric codes but cast to category for modeling convenience)
    df['Culture'] = df['culture'].astype('category')

    # Optional: drop rows with impossible ages (outside 4-14) -- dataset already constrained but keep robust
    df = df[(df['age'] >= 4) & (df['age'] <= 14)]

    # Final dataframe returned contains all columns used in the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a generalized linear model (logistic regression) predicting the probability of choosing the majority option.

    Model specification:
      - Outcome: ChoseMajority (binary)
      - Predictors: Age_c (linear), Age_c2 (quadratic), IsMale, MajorityFirst
      - Culture included as a categorical control and as an interaction with Age_c to allow age slopes to vary across cultures

    Formula used: 'ChoseMajority ~ Age_c + Age_c2 + IsMale + MajorityFirst + C(Culture) + Age_c:C(Culture)'

    Returns the fitted GLMResults object (Binomial family).
    """

    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure the necessary transformed columns exist
    needed = ['ChoseMajority', 'Age_c', 'Age_c2', 'IsMale', 'MajorityFirst', 'Culture']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Fit a logistic regression (GLM with binomial family) with culture fixed effects and age-by-culture interactions
    formula = 'ChoseMajority ~ Age_c + Age_c2 + IsMale + MajorityFirst + C(Culture) + Age_c:C(Culture)'

    glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = glm.fit()

    # Return the fit results object; user can inspect results.summary() or results.params
    return results


