from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/noperturb_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following model columns (exact names used in modeling):
      - MajorityChoice: binary outcome 1 if y==2 (majority), else 0
      - age_c: age centered around the sample mean
      - culture: categorical site identifier (kept as a pandas Categorical)
      - is_female: 1 if gender==1 (girl), 0 otherwise
      - majority_first: binary ordering variable (0/1)

    Drops rows missing any of the required columns.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Drop rows missing any variables required for modeling
    required_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    df = df.dropna(subset=required_cols)

    # Dependent variable: majority choice (y == 2)
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Control: gender -> is_female (1 if girl (gender==1), 0 otherwise)
    # According to dataset code: 1 = girl, 2 = boy
    df['is_female'] = (df['gender'] == 1).astype(int)

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Independent: center age to aid interpretation and numerical stability
    df['age_c'] = df['age'] - df['age'].mean()

    # Culture as categorical factor (keeps original codes but as category)
    df['culture'] = df['culture'].astype('category')

    # Final check: keep only columns needed for the model plus originals if desired
    # (The model will use the columns: MajorityChoice, age_c, culture, is_female, majority_first)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability of choosing the majority option.

    Model specification:
      MajorityChoice ~ age_c * C(culture) + is_female + majority_first

    This specification estimates a main effect of centered age, main effects of culture
    (categorical), and the age-by-culture interaction to test whether developmental
    trajectories (age slopes) differ across cultural contexts. Gender and ordering are
    included as covariates.

    Returns the fitted statsmodels results object (LogitResult).
    """
    import statsmodels.formula.api as smf

    # Ensure the dataframe has the expected columns
    required = ['MajorityChoice', 'age_c', 'culture', 'is_female', 'majority_first']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    formula = 'MajorityChoice ~ age_c * C(culture) + is_female + majority_first'

    # Fit logistic regression (maximum likelihood). Using disp=False to suppress output.
    model = smf.logit(formula=formula, data=df)
    results = model.fit(disp=False)

    # Return the fitted results object. The caller can call results.summary() or
    # extract coefficients, confidence intervals, predicted probabilities, etc.
    return results


