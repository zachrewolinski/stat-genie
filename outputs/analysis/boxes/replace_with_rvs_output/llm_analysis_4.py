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
    """
    Transform the raw dataset into the analysis dataframe. Produces the following new/clean columns used in modeling:
      - MajorityChoice: binary (1 if y==2 (majority), else 0)
      - age_c: centered age (age - mean(age))
      - is_male: 1 if gender==2 (boy), 0 if gender==1 (girl)
      - culture: categorical dtype for site ID
      - majority_first: ensured integer 0/1

    Drops rows with missing values in required columns.
    """
    # work on a copy
    df = df.copy()

    # Required columns for analysis
    required = ['y', 'age', 'culture', 'gender', 'majority_first']

    # Coerce to numeric where appropriate and drop rows with missing values
    for col in ['y', 'age', 'gender', 'majority_first']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert culture to categorical (it may be numeric IDs)
    df['culture'] = df['culture'].astype('category')

    # Drop rows with missing values in required columns after coercion
    df = df.dropna(subset=required)

    # Ensure valid y values (1=unchosen, 2=majority, 3=minority). Keep only these rows
    df = df[df['y'].isin([1, 2, 3])]

    # Create dependent variable: did the child choose the majority option?
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Center age for interpretability and numerical stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Create gender indicator: is_male (1 = boy (gender==2), 0 = girl (gender==1))
    df['is_male'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is integer 0/1 (already 0/1 in schema, but coerce to int)
    df['majority_first'] = df['majority_first'].astype(int)

    # Ensure culture remains categorical and has no unused categories
    df['culture'] = df['culture'].cat.remove_unused_categories()

    # Final drop of any rows that inadvertently got NaNs
    df = df.dropna(subset=['MajorityChoice', 'age_c', 'is_male', 'majority_first', 'culture'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting the binary MajorityChoice outcome.
    The primary predictor is centered age (age_c). Culture is included as a categorical moderator via an interaction (age_c * C(culture)).
    Gender (is_male) and majority_first are included as covariates.

    Model formula:
      MajorityChoice ~ age_c * C(culture) + is_male + majority_first

    Returns the fitted statsmodels results object (LogitResults).
    """
    import statsmodels.formula.api as smf

    # Ensure the culture column is treated as categorical in the formula via C(culture).
    formula = 'MajorityChoice ~ age_c * C(culture) + is_male + majority_first'

    # Fit logistic regression (binomial) using statsmodels' formula API
    # Use disp=False to suppress iterative output in normal usage
    model_res = smf.logit(formula=formula, data=df).fit(disp=False)

    return model_res


