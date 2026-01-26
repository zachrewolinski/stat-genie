from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair dataset into the cleaned dataframe for modeling.

    Produces these additional columns (used in the model):
      - children_binary : 1 if children == 'yes', 0 if 'no'
      - gender_male : 1 if gender == 'male', 0 if 'female'
      - children_gender_interaction : product of children_binary and gender_male
    Also ensures affairs is non-negative integer and drops rows with missing values in model variables.
    """
    df = df.copy()

    # Normalize column names in case of leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Ensure 'affairs' is numeric and non-negative integer
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
    df['affairs'] = df['affairs'].clip(lower=0)
    # Round or cast to integer since the original coding uses integers
    df['affairs'] = df['affairs'].round().astype('Int64')

    # Map children to binary (assumes values 'yes'/'no')
    df['children_binary'] = df['children'].map({
        'yes': 1,
        'no': 0
    })

    # Map gender to male dummy (assumes values 'male'/'female')
    df['gender_male'] = df['gender'].map({
        'male': 1,
        'female': 0
    })

    # Interaction term to allow gender to moderate the effect of children
    df['children_gender_interaction'] = df['children_binary'] * df['gender_male']

    # Ensure numeric controls are numeric
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rows to keep: non-missing in dependent and all predictors/controls
    required_cols = ['affairs', 'children_binary', 'gender_male', 'children_gender_interaction'] + numeric_cols
    # Keep only rows where required columns are not missing
    df = df.dropna(subset=required_cols)

    # Cast columns to appropriate dtypes
    df['children_binary'] = df['children_binary'].astype(int)
    df['gender_male'] = df['gender_male'].astype(int)
    df['children_gender_interaction'] = df['children_gender_interaction'].astype(int)
    df['affairs'] = df['affairs'].astype(int)

    # Reset the index for a clean dataframe
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a Zero-Inflated Negative Binomial (ZINB) model to estimate the effect of having children on the
    number of extramarital affairs. Also provide a (regular) Negative Binomial GLM as a robustness check.

    Returns a dict with keys:
      - 'zinb': fitted ZeroInflatedNegativeBinomialP results
      - 'nb': fitted Negative Binomial GLM results

    Model specification (main eq):
      affairs ~ children_binary + gender_male + children_gender_interaction + age + yearsmarried
                + religiousness + education + occupation + rating

    Inflation (logit) equation (predicting structural zeros):
      children_binary + gender_male + yearsmarried + religiousness
    """
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    import statsmodels.api as sm

    # Select exogenous regressors for count model
    exog_cols = [
        'children_binary',
        'gender_male',
        'children_gender_interaction',
        'age',
        'yearsmarried',
        'religiousness',
        'education',
        'occupation',
        'rating'
    ]

    # Variables to use in the inflation (zero) equation -- a smaller set is reasonable
    exog_infl_cols = [
        'children_binary',
        'gender_male',
        'yearsmarried',
        'religiousness'
    ]

    # Endogenous variable
    endog = df['affairs'].astype(int)

    # Add constant to exog matrices
    exog = sm.add_constant(df[exog_cols], has_constant='add')
    exog_infl = sm.add_constant(df[exog_infl_cols], has_constant='add')

    # Fit Zero-Inflated Negative Binomial (ZINB)
    try:
        zinb_model = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')
        zinb_res = zinb_model.fit(method='bfgs', maxiter=200, disp=False)
    except Exception as e:
        # If ZINB fails to converge, raise a warning and set zinb_res to None
        print('ZINB fitting failed:', e)
        zinb_res = None

    # Fit a (regular) Negative Binomial GLM as a robustness check
    try:
        nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
    except Exception as e:
        print('Negative Binomial GLM fitting failed:', e)
        nb_res = None

    # Print brief summaries for user inspection
    if zinb_res is not None:
        print('Zero-Inflated Negative Binomial results:')
        print(zinb_res.summary())
    if nb_res is not None:
        print('\nNegative Binomial GLM results:')
        print(nb_res.summary())

    # Return results objects for downstream inspection
    return {
        'zinb': zinb_res,
        'nb': nb_res
    }


