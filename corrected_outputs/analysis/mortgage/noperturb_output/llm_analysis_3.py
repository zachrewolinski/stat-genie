from typing import Any
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/noperturb_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston mortgage dataset into a dataframe suitable for logistic regression.

    - Ensures necessary columns exist and are numeric.
    - Drops rows with missing values in any variables used in the model.
    - Creates standardized (z-scored) versions of continuous controls for interpretability.

    Returned dataframe contains the exact column names used in the model:
      ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
       'housing_exp_ratio_std', 'mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std', 'loan_to_value_std']
    """
    df = df.copy()

    # List of raw columns required
    required_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed',
        'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]

    # Ensure required columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Convert columns to numeric where applicable (coerce errors to NaN)
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Ensure binary variables are integers 0/1
    bin_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in bin_cols:
        # round then cast to int to handle floats like 0.0/1.0
        df[c] = df[c].round().astype(int)

    # Standardize continuous controls (z-scoring): create new columns with _std suffix
    cont_map = {
        'housing_expense_ratio': 'housing_exp_ratio_std',
        'mortgage_credit': 'mortgage_credit_std',
        'consumer_credit': 'consumer_credit_std',
        'PI_ratio': 'PI_ratio_std',
        'loan_to_value': 'loan_to_value_std'
    }

    for raw_col, std_col in cont_map.items():
        mean = df[raw_col].mean()
        std = df[raw_col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # if no variation, create zero column
            df[std_col] = 0.0
        else:
            df[std_col] = (df[raw_col] - mean) / std

    # Keep only columns needed for modeling (this is the final dataframe)
    final_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'housing_exp_ratio_std', 'mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std', 'loan_to_value_std'
    ]

    df = df[final_cols].reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (logit) to estimate the effect of gender (female) on mortgage acceptance
    while controlling for applicant and loan characteristics. Returns the fitted model with robust
    (HC0-like) standard errors.

    Model specification:
      accept ~ female + black + self_employed + married + bad_history + denied_PMI
               + housing_exp_ratio_std + mortgage_credit_std + consumer_credit_std +
               PI_ratio_std + loan_to_value_std

    The function prints a summary and returns an object exposing the robust covariance results.
    """
    # Required columns used in the model
    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'housing_exp_ratio_std', 'mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std', 'loan_to_value_std'
    ]

    # Check columns
    missing = [c for c in (['accept'] + X_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    y = df['accept']
    X = df[X_cols]

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression
    logit = sm.Logit(y, X)
    fitted = logit.fit(disp=False)

    # Try to obtain robust covariance results; if not available, compute HC0-like covariance manually
    try:
        # Preferred when available
        fitted_robust = fitted.get_robustcov_results(cov_type='HC0')
        print(fitted_robust.summary())
        return fitted_robust
    except Exception:
        # Manual computation of "HC0" robust covariance for logistic regression (sandwich estimator)
        # Using: cov = inv(H) @ B @ inv(H)
        # where H = X' W X, W = p*(1-p), and B = sum_i (x_i * (y_i - p_i)) (x_i * (y_i - p_i))'
        try:
            # Use the model's stored exog and endog to avoid any mismatch
            X_mat = np.asarray(fitted.model.exog)
            y_obs = np.asarray(fitted.model.endog).reshape(-1)
            # Predicted probabilities
            try:
                p = np.asarray(fitted.predict(X_mat)).reshape(-1)
            except Exception:
                p = np.asarray(fitted.predict()).reshape(-1)

            # Weight for Hessian
            W = p * (1.0 - p)
            # Construct H = X' * diag(W) * X
            Xw = X_mat * W[:, None]
            H = X_mat.T @ Xw

            # Construct B = X' * diag((y-p)^2) * X  (outer product of individual scores)
            resid_diff = (y_obs - p)
            Xr = X_mat * (resid_diff ** 2)[:, None]
            B = X_mat.T @ Xr

            # Invert H; use pseudo-inverse if singular
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = np.linalg.pinv(H)

            cov = H_inv @ B @ H_inv

            # Ensure numeric array
            cov = np.asarray(cov)
            robust_bse = np.sqrt(np.maximum(np.diag(cov), 0.0))

        except Exception:
            # As a last resort, fall back to the model's covariance and standard errors
            cov = np.asarray(fitted.cov_params())
            robust_bse = np.sqrt(np.maximum(np.diag(cov), 0.0))

        class RobustResultsWrapper:
            def __init__(self, base_res, cov_matrix, bse):
                self._base = base_res
                self.params = base_res.params
                self._cov = cov_matrix
                self.bse = bse

            def cov_params(self):
                return self._cov

            def summary(self):
                # Construct a concise table-like string showing robust SEs
                coef = self.params
                se = self.bse
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = coef / se
                pvals = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
                import pandas as _pd
                tbl = _pd.DataFrame({
                    'coef': coef,
                    'std err (HC0)': se,
                    'z': z,
                    'P>|z|': pvals
                })
                return tbl.to_string()

        wrapped = RobustResultsWrapper(fitted, cov, robust_bse)
        print(wrapped.summary())
        return wrapped