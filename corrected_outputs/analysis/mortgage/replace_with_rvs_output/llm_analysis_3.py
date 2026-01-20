from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw HMDA-derived dataframe into the final dataframe used for modeling.

    Steps performed:
    - Make a copy to avoid mutating input.
    - Ensure key columns are numeric and drop rows missing the outcome (accept) or the exposure (female).
    - Ensure continuous covariates are numeric and drop rows missing any of these key predictors.
    - Create z-scored (standardized) versions of continuous predictors for easier coefficient interpretation and numerical stability.
    - Return a dataframe containing only the columns used in the statistical model.
    """
    df = df.copy()

    # Columns expected in input
    expected_cols = [
        'female', 'accept', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]

    # Ensure columns exist (if not, this will raise a KeyError which signals mismatch between schema and data)
    missing = [c for c in expected_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing expected columns in input dataframe: {missing}")

    # Drop rows with missing outcome or main exposure
    df = df.dropna(subset=['accept', 'female'])

    # Convert binary indicators to numeric floats (coerce errors -> NaN)
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Continuous / ordinal predictors
    cont_cols = ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit']
    for c in cont_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing any of the continuous predictors (these are important controls)
    df = df.dropna(subset=cont_cols)

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for stability.
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # If std is zero (unlikely), set z to 0 to avoid division by zero
        if std == 0 or pd.isna(std):
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Final columns to keep for modeling (names must match those used in the model function)
    final_cols = [
        'accept',
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'housing_expense_ratio_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'mortgage_credit_z',
        'consumer_credit_z'
    ]

    # Return only the final columns (preserves row index)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression model to estimate the effect of applicant gender on loan acceptance,
    controlling for applicant credit and financial characteristics.

    Model specification (logit):
      accept ~ female + black + self_employed + married + bad_history + denied_PMI
               + housing_expense_ratio_z + PI_ratio_z + loan_to_value_z + mortgage_credit_z + consumer_credit_z

    Returns:
      A fitted statsmodels Logit results object.
    """
    # Copy to avoid side-effects
    df = df.copy()

    # Define predictor columns (must match the transformed dataframe)
    X_cols = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'housing_expense_ratio_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'mortgage_credit_z',
        'consumer_credit_z'
    ]

    # Ensure predictors are present
    for c in X_cols + ['accept']:
        if c not in df.columns:
            raise KeyError(f"Column required for modeling not found in dataframe: {c}")

    # Prepare design matrix with intercept
    X = sm.add_constant(df[X_cols], has_constant='add')
    y = df['accept']

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    results = logit.fit(disp=False)

    # Return the fitted results object (contains coefficients, std errors, p-values, etc.)
    return results


