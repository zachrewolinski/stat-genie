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
    Prepare and clean the Boston mortgage dataset for modeling.

    Steps:
    - Keep only columns required for the analysis.
    - Coerce to numeric and drop rows with missing values in any required column.
    - Standardize continuous predictors to z-scores (mean 0, sd 1) for interpretability and stability.
    - Ensure binary indicators are integer typed.

    Returns a dataframe containing the final columns used in modeling:
    ['accept','female','black','self_employed','married','bad_history',
     'mortgage_credit_z','consumer_credit_z','PI_ratio_z','loan_to_value_z','housing_expense_ratio_z','denied_PMI']
    """
    df = df.copy()

    # Columns needed for the analysis
    required_cols = [
        'accept',            # dependent variable
        'female',            # independent variable
        'black',
        'self_employed',
        'married',
        'bad_history',
        'mortgage_credit',
        'consumer_credit',
        'PI_ratio',
        'loan_to_value',
        'housing_expense_ratio',
        'denied_PMI'
    ]

    # Subset to required columns (if any are missing this will raise a KeyError so we coerce safely below)
    present = [c for c in required_cols if c in df.columns]
    if len(present) < len(required_cols):
        missing = set(required_cols) - set(present)
        raise KeyError(f"The dataframe is missing required columns: {missing}")

    df = df[required_cols].copy()

    # Coerce to numeric and set non-convertible to NaN
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any of the required predictors/outcome
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Standardize continuous controls (z-score). Use population std (ddof=0) for stability.
    continuous = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in continuous:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # If std is zero (rare), set z to 0 to avoid division by zero
        if std == 0 or np.isnan(std):
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Ensure binary indicators are integers 0/1
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        # round values then cast to int to handle floats like 0.0/1.0
        df[c] = df[c].round().astype(int)

    # Keep only columns needed for modeling (drop the raw continuous columns if desired)
    final_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'denied_PMI'
    ]
    df = df[final_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to estimate the effect of gender (female) on mortgage acceptance
    while controlling for applicant characteristics.

    Model: logit(P(accept=1)) = beta0 + beta1*female + sum(beta_k * controls_k)

    Returns:
    - statsmodels fitted Logit results object
    """
    df = df.copy()

    # Define regressors (controls + main predictor)
    X_cols = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z',
        'denied_PMI'
    ]

    # Check that required columns are present
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for modeling: {missing}")

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    # use lbfgs for stability and suppress iterative output
    results = logit_model.fit(disp=False, method='lbfgs', maxiter=200)

    # Print a concise summary for quick inspection (caller can inspect returned results object for details)
    print(results.summary())

    return results


