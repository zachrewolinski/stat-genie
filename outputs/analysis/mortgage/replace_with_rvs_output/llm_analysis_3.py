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
    - Ensure relevant columns are numeric and drop rows with missing values on outcome, key IV, and controls.
    - Create standardized (z-scored) versions of continuous controls to aid interpretation and numerical stability.
    - Returns dataframe containing all columns listed in the conceptual variables.
    """
    # Keep a copy to avoid modifying input in place
    df = df.copy()

    # Columns required for the analysis
    required_cols = [
        'accept',
        'female',
        'black',
        'married',
        'self_employed',
        'bad_history',
        'denied_PMI',
        'mortgage_credit',
        'consumer_credit',
        'PI_ratio',
        'loan_to_value',
        'housing_expense_ratio'
    ]

    # Coerce these columns to numeric where appropriate
    for col in required_cols:
        if col in df.columns:
            # errors='coerce' will turn non-convertible values into NaN which will be dropped below
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise KeyError(f"Required column '{col}' not found in dataframe")

    # Drop rows with missing values in the required columns
    df = df.dropna(subset=required_cols)

    # Ensure binary indicators are integer 0/1
    binary_cols = ['accept', 'female', 'black', 'married', 'self_employed', 'bad_history', 'denied_PMI']
    for col in binary_cols:
        # round then clip to {0,1}
        df[col] = df[col].round().astype(int).clip(0, 1)

    # Standardize continuous controls (z-scores). Use population std (ddof=0) for stability.
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in cont_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, create zero column to avoid divide-by-zero
            df['z_' + col] = 0.0
        else:
            df['z_' + col] = (df[col] - mean) / std

    # Final column list that will be used by the model (keeps original binary columns and z_ continuous)
    final_cols = [
        'accept',
        'female',
        'black',
        'married',
        'self_employed',
        'bad_history',
        'denied_PMI',
        'z_mortgage_credit',
        'z_consumer_credit',
        'z_PI_ratio',
        'z_loan_to_value',
        'z_housing_expense_ratio'
    ]

    # Subset to final columns (keeps dataframe compact). This also ensures the exact column names exist.
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression for the binary outcome 'accept' on 'female' plus controls.
    Returns a dict with the fitted model result object and average marginal effects.

    Model specification (logistic):
      accept ~ female + black + married + self_employed + bad_history + denied_PMI
               + z_mortgage_credit + z_consumer_credit + z_PI_ratio + z_loan_to_value + z_housing_expense_ratio

    Notes:
    - We include a constant in the model.
    - We compute average marginal effects (AME) to make interpretation on probability scale straightforward.
    """
    # Copy to avoid side-effects
    df = df.copy()

    # Define predictors used in the model (must match transform output column names)
    predictors = [
        'female',
        'black',
        'married',
        'self_employed',
        'bad_history',
        'denied_PMI',
        'z_mortgage_credit',
        'z_consumer_credit',
        'z_PI_ratio',
        'z_loan_to_value',
        'z_housing_expense_ratio'
    ]

    # Ensure no missing values (transform should have removed them already)
    df = df.dropna(subset=['accept'] + predictors)

    # Prepare design matrices
    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression using statsmodels Logit; use robust covariance (HC1) for standard errors
    logit_model = sm.Logit(y, X)
    try:
        result = logit_model.fit(disp=False, cov_type='HC1')
    except TypeError:
        # Some versions of statsmodels don't accept cov_type in fit(...). Fit then get robust cov separately.
        result = logit_model.fit(disp=False)
        # attempt to get robust covariance if available
        try:
            robust_res = result.get_robustcov_results(cov_type='HC1')
            result = robust_res
        except Exception:
            # fall back to the original result
            pass

    # Compute average marginal effects (AME) for interpretation on probability scale
    try:
        marg = result.get_margeff(at='overall', method='dydx')
        marg_summary = marg.summary_frame()
    except Exception:
        marg_summary = None

    # Return the fitted result and marginal effects table (pandas DataFrame) if available
    return {
        'model_result': result,
        'average_marginal_effects': marg_summary
    }


