from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations performed:
    - Keep a copy of the dataframe.
    - Drop rows with missing values in any columns required for the model.
    - Coerce relevant columns to numeric and drop rows that become NA after coercion.
    - Create z-scored (standardized) versions of continuous/ordinal predictors used as controls:
      mortgage_credit_z, consumer_credit_z, PI_ratio_z, loan_to_value_z, housing_expense_ratio_z.
    - Ensure binary indicator columns are integer-typed.

    Returns the transformed dataframe containing at least the columns used in the model.
    """
    df = df.copy()

    # Columns required for modeling
    required_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio'
    ]

    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Coerce to numeric where appropriate (safe guard if strings present)
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows that could not be coerced
    df = df.dropna(subset=required_cols)

    # Standardize continuous / ordinal predictors (z-score). Use population std (ddof=0) for stability.
    to_z = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in to_z:
        zcol = f"{c}_z"
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # If std is zero (unlikely), set z to zero to avoid division by zero
        if std == 0 or np.isnan(std):
            df[zcol] = 0.0
        else:
            df[zcol] = (df[c] - mean) / std

    # Ensure binary indicators are integer type (0/1)
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history']
    for c in binary_cols:
        df[c] = df[c].astype(int)

    # Return dataframe containing all columns needed by the model
    # (original binary columns plus standardized controls)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression to estimate the association between applicant gender and mortgage acceptance.

    Model specification (primary):
      accept ~ female + black + self_employed + married + bad_history
               + mortgage_credit_z + consumer_credit_z + PI_ratio_z + loan_to_value_z + housing_expense_ratio_z

    - Uses statsmodels.Logit; if Logit fails to converge or faces issues, falls back to GLM with Binomial family.
    - Returns the fitted results object (statsmodels results instance).
    """
    df = df.copy()

    # Outcome and predictors used in model (must match columns created in transform)
    y = df['accept']
    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z'
    ]

    X = df[X_cols]

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression; provide robust fall-back if needed
    try:
        logit = sm.Logit(y, X)
        results = logit.fit(disp=False)
    except Exception:
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        results = glm.fit()

    return results


