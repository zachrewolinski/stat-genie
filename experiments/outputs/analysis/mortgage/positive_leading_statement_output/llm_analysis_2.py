from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/positive_leading_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the mortgage dataset for modeling.

    Produces standardized (z-scored) continuous controls and ensures binary variables
    are integer typed. Drops rows with missing values on variables used in the model.

    Final columns added/used in modeling:
      - housing_exp_ratio_z, PI_ratio_z, loan_to_value_z,
        mortgage_credit_z, consumer_credit_z (standardized continuous controls)
      - binary columns kept as integers: accept, female, black, self_employed,
        married, bad_history, denied_PMI
    """
    df = df.copy()

    # Columns required for this analysis
    required_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed',
        'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Ensure binary flags are integer (0/1)
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        # Some columns might be floats (e.g., 0.0/1.0); cast to int
        df[c] = df[c].astype(int)

    # Standardize continuous predictors (z-score). Use sample std (ddof=1).
    cont_cols = {
        'housing_expense_ratio': 'housing_exp_ratio_z',
        'PI_ratio': 'PI_ratio_z',
        'loan_to_value': 'loan_to_value_z',
        'mortgage_credit': 'mortgage_credit_z',
        'consumer_credit': 'consumer_credit_z'
    }

    for orig, zname in cont_cols.items():
        mean = df[orig].mean()
        std = df[orig].std(ddof=1)
        # If std is zero for any reason, create zero column to avoid division by zero
        if std == 0 or np.isnan(std):
            df[zname] = 0.0
        else:
            df[zname] = (df[orig] - mean) / std

    # Keep only rows with finite values in z-scored columns (defensive)
    z_cols = list(cont_cols.values())
    df = df[df[z_cols].notnull().all(axis=1)]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (logit) to estimate the effect of applicant gender on
    mortgage acceptance while controlling for applicant characteristics.

    Returns a dictionary with the fitted model object, odds ratios, confidence intervals,
    and a summary table.
    """
    # Defensive copy
    data = df.copy()

    # Define outcome and predictors (must match transform output column names)
    y = data['accept']

    X_cols = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'housing_exp_ratio_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'mortgage_credit_z',
        'consumer_credit_z'
    ]

    X = data[X_cols]

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    # Use try/except to surface convergence issues if they occur
    try:
        logit_model = sm.Logit(y, X)
        fit = logit_model.fit(disp=False)
    except Exception as e:
        # If Logit fails (e.g., perfect separation), try a penalized fit via statsmodels' method
        fit = logit_model.fit_regularized(method='l1', disp=False)

    # Compute odds ratios and confidence intervals on the original scale
    params = fit.params
    conf = fit.conf_int()
    conf.columns = ['2.5%', '97.5%']

    odds_ratios = np.exp(params)
    conf_exp = np.exp(conf)

    results = {
        'model_fit': fit,
        'odds_ratios': odds_ratios,
        'conf_int_exp': conf_exp,
        'pvalues': fit.pvalues,
        'summary': fit.summary()  # a textual summary object
    }

    return results


