from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for modeling the effect of having children on extramarital affairs.

    Produces the following columns used for modeling:
      - Affairs: numeric outcome (from original 'affairs')
      - HasChildren: binary (1=yes, 0=no) derived from 'children'
      - GenderMale: binary (1=male, 0=female) derived from 'gender'
      - Age: numeric (from 'age')
      - YearsMarried: numeric (from 'yearsmarried')
      - Religiousness: numeric (from 'religiousness')
      - Education: numeric (from 'education')
      - Occupation: numeric (from 'occupation')
      - MarriageRating: numeric (from 'rating')

    Rows with missing values in any of the above required fields are dropped.
    """
    df = df.copy()

    # Ensure relevant raw columns are present
    required_raw = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    # Drop rows missing any of the required raw fields
    df = df.dropna(subset=required_raw)

    # Dependent variable: keep original numeric coding
    df['Affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Independent: children -> binary indicator
    # Accept variations like 'yes', 'Yes', 'no', 'No'; default to 0 if not clearly 'yes'
    df['HasChildren'] = df['children'].astype(str).str.strip().str.lower().apply(lambda x: 1 if x.startswith('y') else 0)

    # Control: gender -> binary male indicator (1 male, 0 female)
    df['GenderMale'] = df['gender'].astype(str).str.strip().str.lower().apply(lambda x: 1 if x.startswith('m') else 0)

    # Numeric controls: coerce to numeric, invalid parsing becomes NaN and will be dropped
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['MarriageRating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop any rows with NA in the transformed variables
    model_cols = ['Affairs', 'HasChildren', 'GenderMale', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'MarriageRating']
    df = df.dropna(subset=model_cols)

    # Return dataframe that contains the exact columns used by the statistical model
    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit primary and robustness models to estimate the association between having children and engagement in extramarital affairs.

    Primary specification: Zero-Inflated Negative Binomial (ZINB) to account for the count nature of 'Affairs' and the excess zeros.
    Robustness: OLS with heteroskedasticity-robust SEs.

    Returns a dictionary with fitted result objects:
      - 'zinb': fitted ZeroInflatedNegativeBinomialP results
      - 'ols_robust': fitted OLS results with HC3 robust SEs
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Ensure a copy so the caller's dataframe is not modified
    df = df.copy()

    # Define regressors
    exog_vars = ['HasChildren', 'GenderMale', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'MarriageRating']
    # Exog for count component
    exog = sm.add_constant(df[exog_vars])
    # Exog for inflation (logit) component: include a smaller set of demographics likely to predict always-zero (no affairs)
    exog_infl = sm.add_constant(df[['HasChildren', 'GenderMale', 'Age', 'YearsMarried', 'Religiousness']])

    endog = df['Affairs']

    # Fit Zero-Inflated Negative Binomial (primary)
    try:
        zinb_mod = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')
        zinb_res = zinb_mod.fit(method='newton', maxiter=100, disp=False)
    except Exception as e:
        # If fitting fails, return the exception message in place of results
        zinb_res = e

    # Robust OLS as a simple comparison/robustness check
    ols_mod = sm.OLS(endog, exog)
    ols_res = ols_mod.fit(cov_type='HC3')

    return {'zinb': zinb_res, 'ols_robust': ols_res}


