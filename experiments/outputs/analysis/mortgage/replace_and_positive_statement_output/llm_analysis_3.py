from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet, Literal
import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import scipy  # noqa: F401
import statsmodels.api as sm
import statsmodels.formula.api as smf  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import pickle  # noqa: F401
from statsmodels.stats.sandwich_covariance import cov_hc3

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_and_positive_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Boston Fed mortgage dataset for modeling.

    Produces standardized continuous covariates and a binary accept variable named 'accept_binary'.
    Drops rows with missing values in the variables required for the primary model.
    """
    df = df.copy()

    # Required columns for the analysis (raw input names)
    required_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value',
        'housing_expense_ratio', 'denied_PMI'
    ]

    # Ensure columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing values in any required column (listwise deletion for this model)
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Dependent variable: make sure accept is binary and create accept_binary
    # In dataset 'accept' is 1 if accepted, 0 if denied; coerce to int
    df['accept_binary'] = df['accept'].astype(int)

    # Ensure key binary controls are integer-coded 0/1
    binary_cols = ['female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for b in binary_cols:
        df[b] = df[b].astype(int)

    # Standardize continuous predictors to make coefficients comparable and improve numerical stability
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # If std is zero (unlikely), set standardized value to zero to avoid division by zero
        if std == 0 or np.isnan(std):
            df[c + '_std'] = 0.0
        else:
            df[c + '_std'] = (df[c] - mean) / std

    # Final dataframe includes original variables plus the standardized versions and accept_binary
    # Keep relevant columns for modeling (but return full df copy so user can inspect other fields)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to estimate the effect of applicant gender on mortgage approval.

    Returns a dictionary containing a results-wrapper object with a robust covariance matrix attached
    and a marginal effects summary for the female indicator.
    """
    df = df.copy()

    # Ensure transform has been applied (accept_binary and standardized columns exist)
    needed = [
        'accept_binary', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std', 'loan_to_value_std', 'housing_expense_ratio_std'
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required transformed columns for modeling: {missing}")

    # Define model covariates
    X_cols = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'mortgage_credit_std',
        'consumer_credit_std',
        'PI_ratio_std',
        'loan_to_value_std',
        'housing_expense_ratio_std',
        'denied_PMI'
    ]

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept_binary']

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    try:
        res = logit_model.fit(disp=False, maxiter=100)
    except Exception:
        # Try a more robust solver if default has issues
        res = logit_model.fit(disp=False, method='bfgs', maxiter=200)

    # Compute robust (heteroskedasticity-consistent) covariance matrix (HC3)
    try:
        robust_cov = cov_hc3(res)
    except Exception:
        # Fallback: use the model-based covariance if sandwich estimator fails for some reason
        robust_cov = res.cov_params()

    # Create a lightweight wrapper object that exposes the original result and robust covariance
    class RobustResultWrapper:
        def __init__(self, original_res, cov_matrix):
            self._res = original_res
            self.cov_robust = cov_matrix
            # Expose params and conventional bse for convenience
            self.params = getattr(original_res, 'params', None)
            # robust standard errors
            try:
                self.bse_robust = np.sqrt(np.diag(cov_matrix))
            except Exception:
                self.bse_robust = None

        def __getattr__(self, name):
            # Delegate attribute access to the original results object
            return getattr(self._res, name)

        def summary(self, *args, **kwargs):
            # Prefer the original summary
            return self._res.summary(*args, **kwargs)

    res_robust_wrapper = RobustResultWrapper(res, robust_cov)

    # Compute average marginal effects (AME) to give a straightforward interpretation of the female effect
    margeff_summary = None
    try:
        # Try requesting robust cov_type for marginal effects if supported
        try:
            margeff = res.get_margeff(at='overall', method='dydx', cov_type='HC3')
        except TypeError:
            # Older/newer versions may not accept cov_type; fall back
            margeff = res.get_margeff(at='overall', method='dydx')
        except Exception:
            # Any other failure, attempt without cov_type
            margeff = res.get_margeff(at='overall', method='dydx')

        # summary() typically returns a statsmodels.iolib.summary.Summary object
        margeff_summary = margeff.summary()
    except Exception:
        # If marginal effects computation fails, leave summary as None
        margeff_summary = None

    # Return results. The 'logit_robust_result' entry is a wrapper exposing original result and robust covariance.
    return {
        'logit_robust_result': res_robust_wrapper,
        'marginal_effects_summary': margeff_summary
    }