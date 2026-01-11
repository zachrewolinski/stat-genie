from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/examples/mortgage/analysis5_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for logistic regression of loan acceptance on gender with controls.

    Produces standardized continuous controls and drops rows with missing values in any variables used.

    Final returned dataframe contains the exact columns referenced in the model:
    ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
     'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z']
    """
    # work on a copy
    df = df.copy()

    # Columns required (original names in provided schema)
    required_orig = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio'
    ]

    # Ensure required columns exist
    missing = [c for c in required_orig if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Coerce numeric columns to numeric, converting non-parsable entries to NaN
    for c in required_orig:
        # keep binary columns as numeric too
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any variable we will use
    df = df.dropna(subset=required_orig)

    # Standardize continuous controls (z-score). Use sample std (ddof=1) to mirror common practice.
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std()
        if std == 0 or np.isnan(std):
            # If constant or degenerate, create zero column
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Final columns for modeling
    final_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z'
    ]

    # Return only the columns needed for modeling (cleaned and standardized)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits a logistic regression model predicting loan acceptance (accept) from female and controls.

    Model specification (logit):
      accept ~ female + black + self_employed + married + bad_history
               + mortgage_credit_z + consumer_credit_z + PI_ratio_z + loan_to_value_z + housing_expense_ratio_z

    Returns the fitted statsmodels results object (LogitResults).
    """
    # Work on a copy
    df = df.copy()

    # Ensure required columns are present
    required = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z'
    ]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns in transformed dataframe: {missing}")

    # Define outcome and predictors
    y = df['accept'].astype(float)
    X = df[[
        'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z'
    ]]

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Return fitted results object for downstream inspection (coefficients, p-values, summary, etc.)
    return results


