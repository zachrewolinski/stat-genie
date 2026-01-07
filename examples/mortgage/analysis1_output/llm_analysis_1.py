from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/mortgage/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for modeling the effect of gender on mortgage approval.

    Transformations performed:
    - Create binary dependent variable 'accepted' from 'accept' (or from 'deny' if 'accept' is missing).
    - Ensure key binary columns are integer 0/1.
    - Drop rows with missing values in DV, IV, and required control variables.
    - Standardize continuous control variables and store as *_z columns.

    Final dataframe will include the columns referenced in the conceptual variables:
    'accepted', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
    'PI_ratio_z', 'loan_to_value_z', 'bad_history', 'married', 'self_employed',
    'housing_expense_ratio_z'
    """
    df = df.copy()

    # Construct dependent variable 'accepted' from available columns
    if 'accept' in df.columns:
        df['accepted'] = df['accept']
    elif 'deny' in df.columns:
        # if only 'deny' is present, infer acceptance
        df['accepted'] = 1 - df['deny']
    else:
        raise KeyError("Dataset must contain either 'accept' or 'deny' column to determine outcome.")

    # Required binary columns must exist; do not cast to int yet because there may be missing values
    binary_cols = ['female', 'black', 'bad_history', 'married', 'self_employed']
    for col in binary_cols:
        if col not in df.columns:
            # if a required binary control is missing, raise to make user aware
            raise KeyError(f"Required column '{col}' not found in dataframe.")

    # Required continuous controls
    required_continuous = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in required_continuous:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe.")

    # Drop rows with missing values in DV, IV, and controls
    required_for_model = ['accepted', 'female', 'black', 'mortgage_credit', 'consumer_credit',
                          'PI_ratio', 'loan_to_value', 'bad_history', 'married', 'self_employed',
                          'housing_expense_ratio']
    df = df.dropna(subset=required_for_model)

    # After dropping rows with missing required values, it's safe to cast binaries and accepted to integer 0/1
    # This will fail if values are non-numeric strings; we preserve original semantics (expecting numeric/bool)
    for col in binary_cols:
        # convert booleans or floats to integer 0/1
        df[col] = df[col].astype(int)
    df['accepted'] = df['accepted'].astype(int)

    # Standardize continuous variables (z-scores) and store as new columns
    cont_to_z = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in cont_to_z:
        # use population std (ddof=0) for standardization; ddof=1 could also be used
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If constant or invalid, set z to 0 to avoid division by zero
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Final check: ensure all model columns exist
    model_cols = ['accepted', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
                  'PI_ratio_z', 'loan_to_value_z', 'bad_history', 'married', 'self_employed',
                  'housing_expense_ratio_z']
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after transform: {missing}")

    # Return transformed dataframe (keeps other columns as well if user wants them)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression model predicting mortgage acceptance from applicant gender
    controlling for credit- and loan-related covariates.

    The model specification (in matrix form):
      logit(P(accepted=1)) = beta0 + beta1*female + beta2*black + beta3*mortgage_credit_z + ...

    Returns a dictionary with the fitted statsmodels LogitResults object and a small
    summary table of odds ratios with 95% CIs and p-values.
    """
    # Required model columns (must match columns created in transform)
    X_cols = ['female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
              'PI_ratio_z', 'loan_to_value_z', 'bad_history', 'married', 'self_employed',
              'housing_expense_ratio_z']

    # Check columns present
    missing = [c for c in X_cols + ['accepted'] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for modeling: {missing}")

    # Prepare design matrices
    X = df[X_cols]
    X = sm.add_constant(X)
    y = df['accepted']

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    # suppress iterative output with disp=False
    results = logit_model.fit(disp=False)

    # Create odds ratio table with 95% CI and p-values
    params = results.params
    conf = results.conf_int()
    or_table = pd.DataFrame({
        'coef': params,
        'odds_ratio': np.exp(params),
        'ci_lower': np.exp(conf.iloc[:, 0]),
        'ci_upper': np.exp(conf.iloc[:, 1]),
        'p_value': results.pvalues
    })

    # Return results: fitted model and odds ratio table for easy interpretation
    return {
        'results': results,
        'odds_ratio_table': or_table
    }