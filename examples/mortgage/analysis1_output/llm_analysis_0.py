from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import scipy  # noqa: F401
import statsmodels.api as sm
import statsmodels.formula.api as smf  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import pickle  # noqa: F401
import types

# Load dataset (path left as in original; consumers of this module may replace as needed)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/mortgage/data.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe into the final dataframe used for modeling.

    Steps:
    - Make a copy of the input.
    - Ensure key columns exist and drop rows with missing values in any variable used in the model.
    - Coerce types for binary indicators to numeric (then integer after dropping coercion-induced NA).
    - Coerce numeric columns to numeric and drop rows with non-numeric or missing values.
    - Create standardized (z-scored) versions of continuous predictors for easier interpretation and numeric stability.

    Final dataframe columns used by the model:
      'accept', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
      'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
      'self_employed', 'married', 'bad_history'
    """
    df = df.copy()

    # Columns required for modeling (original names)
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'self_employed', 'married', 'bad_history'
    ]

    # Keep only rows with non-missing values in these columns (initial filter)
    df = df.dropna(subset=required_cols)

    # Ensure continuous predictors are numeric (coerce non-numeric to NaN)
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in cont_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure binary/numeric indicator columns are numeric (coerce non-numeric to NaN).
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history']
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that became missing after coercion
    df = df.dropna(subset=cont_cols + binary_cols)

    # Now it's safe to cast binary columns to integer (0/1). Use int64.
    for col in binary_cols:
        # Some numeric values might be floats like 0.0/1.0; this cast is safe after dropna
        df[col] = df[col].astype(int)

    # Standardize continuous predictors (z-scores). Use population std (ddof=0).
    for col in cont_cols:
        zcol = f"{col}_z"
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # guard against zero std
        if std == 0 or pd.isna(std):
            df[zcol] = 0.0
        else:
            df[zcol] = (df[col] - mean) / std

    # Final columns kept (explicit) - note: we keep original continuous columns too, but model will use _z columns
    final_cols = [
        'accept', 'female', 'black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'self_employed', 'married', 'bad_history'
    ]

    # Ensure final z columns exist (they do from above). Return the dataframe (may contain other columns).
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting mortgage approval (accept) from applicant gender
    controlling for creditworthiness and demographic/business variables.

    Model specification (logistic):
      accept ~ female + black + mortgage_credit_z + consumer_credit_z + PI_ratio_z
               + loan_to_value_z + housing_expense_ratio_z + self_employed + married + bad_history

    Returns the fitted statsmodels results object with robust standard errors (HC3) when possible.
    """
    # Make a local copy
    df = df.copy()

    # Define model predictors (these must exist in the transformed dataframe)
    X_cols = [
        'female', 'black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'self_employed', 'married', 'bad_history'
    ]

    # Ensure the required columns are present
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError('Missing required columns in dataframe for modeling: ' + ','.join(missing))

    # Design matrices
    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['accept'].astype(float)

    # Fit logistic regression (maximum likelihood). Use disp=False to suppress fit output.
    logit = sm.Logit(y, X)

    # First attempt: fit normally
    results = logit.fit(disp=False)

    # Try to obtain a results object with HC3 robust covariance.
    # Different versions of statsmodels expose different APIs:
    # - Some have get_robustcov_results on results
    # - Some accept cov_type in fit(...)
    # We'll attempt both, and if unavailable fall back to computing HC3 covariance and monkey-patching.
    try:
        if hasattr(results, 'get_robustcov_results'):
            results_robust = results.get_robustcov_results(cov_type='HC3')
        else:
            # Attempt to re-fit requesting robust cov directly
            try:
                results_robust = logit.fit(disp=False, cov_type='HC3')
            except TypeError:
                # Final fallback: compute HC3 covariance and attach it to the results object
                from statsmodels.stats.sandwich_covariance import cov_hc3
                cov = cov_hc3(results)
                # Monkey-patch cov_params method to return the robust covariance
                def cov_params_robust(self, *args, **kwargs):
                    return cov
                results.cov_params = types.MethodType(cov_params_robust, results)
                # Update bse to reflect robust standard errors
                try:
                    results.bse = np.sqrt(np.diag(cov))
                except Exception:
                    # If diag fails for some reason, leave bse as-is
                    pass
                # Also store cov_robust for external inspection
                results.cov_robust = cov
                results_robust = results
    except Exception:
        # If anything unexpected fails, fall back to the original results (non-robust)
        results_robust = results

    return results_robust