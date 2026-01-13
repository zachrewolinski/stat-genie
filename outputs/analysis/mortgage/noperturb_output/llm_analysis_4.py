from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/noperturb_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw HMDA-style dataframe to a modeling-ready dataframe.

    Steps:
    - Work on a copy of df.
    - Coerce key columns to numeric / integer where appropriate.
    - Drop rows with missing values in any variables used in the model.
    - Standardize continuous ratio variables (PI_ratio, loan_to_value, housing_expense_ratio)
      and store standardized versions as z_PI_ratio, z_loan_to_value, z_housing_expense_ratio.

    Returns the transformed dataframe containing at minimum the columns used in the model:
    'accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI',
    'z_PI_ratio', 'z_loan_to_value', 'z_housing_expense_ratio', 'mortgage_credit', 'consumer_credit'.
    """
    df = df.copy()

    # Columns we will use
    needed_cols = [
        'accept', 'female', 'black', 'bad_history', 'married', 'self_employed',
        'denied_PMI', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'mortgage_credit', 'consumer_credit'
    ]

    # Convert to numeric where appropriate
    for col in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']:
        if col in df.columns:
            # Convert floats of 0/1 to int and coerce non-numeric to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float')

    # Convert credit scores and continuous ratios to numeric
    for col in ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in any of the needed columns
    present_needed = [c for c in needed_cols if c in df.columns]
    df = df.dropna(subset=present_needed)

    # Now, cast binary columns to int
    for col in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Standardize continuous ratio covariates and create z_ prefixed columns
    for col in ['PI_ratio', 'loan_to_value', 'housing_expense_ratio']:
        zcol = 'z_' + col
        if col in df.columns:
            # population std (ddof=0) for scale stability; skip constant columns
            std = df[col].std(ddof=0)
            if std == 0 or np.isclose(std, 0):
                df[zcol] = 0.0
            else:
                df[zcol] = (df[col] - df[col].mean()) / std

    # Ensure mortgage_credit and consumer_credit are integers if they represent ordered score categories
    for col in ['mortgage_credit', 'consumer_credit']:
        if col in df.columns:
            # round if they are float-coded categories, then cast to int
            df[col] = df[col].round().astype(int)

    # Keep only the columns needed for modeling (this ensures the final dataframe contains exactly the columns referenced)
    final_cols = [
        'accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI',
        'z_PI_ratio', 'z_loan_to_value', 'z_housing_expense_ratio', 'mortgage_credit', 'consumer_credit'
    ]
    # Filter to existing columns among final_cols (in case some original columns were absent)
    final_cols_present = [c for c in final_cols if c in df.columns]
    df = df[final_cols_present].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a multivariate logistic regression (logit) predicting loan acceptance from gender
    while adjusting for applicant and loan characteristics.

    Model:
      accept ~ female + black + bad_history + married + self_employed + denied_PMI
               + z_PI_ratio + z_loan_to_value + z_housing_expense_ratio
               + mortgage_credit + consumer_credit

    Returns the fitted statsmodels Logit results object.
    """
    # Required columns for the model (must match transform output names)
    X_cols = [
        'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI',
        'z_PI_ratio', 'z_loan_to_value', 'z_housing_expense_ratio',
        'mortgage_credit', 'consumer_credit'
    ]

    # Ensure required columns are present
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    X = df[X_cols].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (use GLM or Logit). Using Logit for standard MLE logistic regression.
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # For interpretation, attach odds ratios and 95% CIs as attributes on the results (optional)
    try:
        params = results.params
        conf = results.conf_int()
        or_vals = np.exp(params)
        or_ci = np.exp(conf)
        results.odds_ratios = or_vals
        results.odds_ratios_ci = or_ci
    except Exception:
        # If something goes wrong computing ORs, ignore and return results anyway
        pass

    return results


