from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe for modeling majority reliance as a function of age and culture.

    Produces the following added columns used in the model:
      - ChoseMajority: binary (1 if y==2, else 0)
      - age_c: age centered around sample mean
      - age_c_sq: (age_c)**2 to capture nonlinearity
      - is_boy: indicator 1 if gender == 2 (boy), 0 if gender == 1 (girl)
    Ensures majority_first is integer (0/1) and drops rows missing required fields.
    """
    df = df.copy()

    # Required columns: y, age, culture, gender, majority_first
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=required)

    # Dependent variable: did the child choose the majority option? (y == 2)
    df['ChoseMajority'] = (df['y'] == 2).astype(int)

    # Center age and add quadratic term to allow for nonlinear development
    df['age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()
    df['age_c_sq'] = df['age_c'] ** 2

    # Gender: make explicit binary indicator for boy (gender: 1=girl, 2=boy)
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is binary integer
    df['majority_first'] = df['majority_first'].astype(int)

    # Keep only columns needed for modeling (but return full df with added cols)
    # Note: model function will handle categorical coding of 'culture' via formula C(culture)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a logistic regression predicting whether a child chose the majority option.

    Model specification:
      ChoseMajority ~ age_c + age_c_sq + C(culture) + age_c:C(culture) + is_boy + majority_first

    The age_c:C(culture) interaction tests whether the developmental slope (change with age)
    differs across cultural sites. We report cluster-robust standard errors clustered by culture
    to account for within-site dependence.

    Returns the fitted results object with cluster-robust covariance.
    """
    import statsmodels.formula.api as smf

    # Ensure necessary columns exist
    needed = ['ChoseMajority', 'age_c', 'age_c_sq', 'culture', 'is_boy', 'majority_first']
    if not all(col in df.columns for col in needed):
        raise ValueError('Dataframe is missing required columns for the model. Run transform() first.')

    # Formula including interaction between centered age and culture (culture treated as categorical)
    formula = 'ChoseMajority ~ age_c + age_c_sq + C(culture) + age_c:C(culture) + is_boy + majority_first'

    # Fit logistic regression (maximum likelihood)
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Obtain cluster-robust standard errors clustered by culture
    # (useful because observations within cultural sites may not be independent)
    try:
        results = model.get_robustcov_results(cov_type='cluster', groups=df['culture'])
    except Exception:
        # If clustering fails, fall back to default (non-clustered) results
        results = model

    return results


