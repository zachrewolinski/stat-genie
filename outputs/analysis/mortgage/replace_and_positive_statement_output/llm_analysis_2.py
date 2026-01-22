from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_and_positive_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Boston mortgage dataset for logistic regression.

    Steps performed:
    - Copy dataframe to avoid modifying original.
    - Drop the (unused) index column 'Unnamed: 0' if present.
    - Drop rows with missing values in any columns required for the model.
    - Ensure binary columns are integers.
    - Standardize continuous predictors to mean 0, sd 1 (suffix '_s').

    Returns the transformed dataframe containing all columns used by the model.
    """
    df = df.copy()

    # drop index-like column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Columns required for analysis
    required_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed',
        'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]

    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Ensure binary/int columns are integer type
    for bin_col in ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']:
        # cast safely
        df[bin_col] = df[bin_col].astype(int)

    # Standardize continuous predictors (create new columns with suffix _s)
    cont_cols = ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit']
    for c in cont_cols:
        # compute mean/std on available rows
        mean_c = df[c].mean()
        std_c = df[c].std()
        # Avoid division by zero
        if std_c == 0 or np.isnan(std_c):
            df[c + '_s'] = 0.0
        else:
            df[c + '_s'] = (df[c] - mean_c) / std_c

    # Final dataframe contains both original and standardized columns; modeling uses standardized versions
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (logit) predicting loan acceptance from applicant gender
    controlling for creditworthiness, demographics, and loan characteristics.

    Returns a dict containing:
      - 'model': the fitted statsmodels LogitResults object
      - 'odds_ratios': a DataFrame with coefficients, standard errors, ORs and 95% CI

    The key test is the coefficient on 'female' (if OR significantly different from 1,
    gender affects approval probability controlling for covariates).
    """
    df = df.copy()

    # Predictor variables (female is the independent variable of interest)
    X_vars = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'housing_expense_ratio_s',
        'PI_ratio_s',
        'loan_to_value_s',
        'mortgage_credit_s',
        'consumer_credit_s'
    ]

    # Ensure the model columns exist in df
    missing = [c for c in X_vars + ['accept'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    X = df[X_vars]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression
    logit_model = sm.Logit(y, X)
    # disp=False to avoid printing optimization output
    res = logit_model.fit(disp=False)

    # Build odds ratios table with 95% CI
    coef = res.params
    se = res.bse
    ci = res.conf_int()
    ci.columns = ['ci_lower', 'ci_upper']

    or_table = pd.DataFrame({
        'coef': coef,
        'se': se,
        'OR': np.exp(coef),
        'OR_ci_lower': np.exp(ci['ci_lower']),
        'OR_ci_upper': np.exp(ci['ci_upper'])
    })

    # Return fitted model and a tidy odds-ratio table for interpretation
    return {
        'model': res,
        'odds_ratios': or_table
    }


