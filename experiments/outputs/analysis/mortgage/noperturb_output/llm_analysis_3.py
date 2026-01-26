from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as scistats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# NOTE: This top-level read mirrors the original file; it may be unused during tests
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/noperturb_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling. This function:
    - keeps only columns required for the analysis
    - coerces types where appropriate
    - drops rows with missing values in any required column
    - ensures binary columns are integer type
    Returns the cleaned dataframe containing the exact column names used in the model.
    """
    df = df.copy()

    # Columns required for the analysis (IV, DV, and controls)
    required_cols = [
        'female',
        'accept',
        'black',
        'housing_expense_ratio',
        'self_employed',
        'married',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'denied_PMI'
    ]

    # Keep only the required columns if they exist in the incoming dataframe
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataframe is missing required columns: {missing_cols}")

    df = df[required_cols]

    # Coerce numeric columns to numeric types (will produce NaN for non-convertible values)
    numeric_cols = [
        'housing_expense_ratio',
        'mortgage_credit',
        'consumer_credit',
        'PI_ratio',
        'loan_to_value'
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure binary indicator columns are numeric (coerce and then drop NA below)
    binary_cols = ['female', 'accept', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        # Some datasets may have these as float/str; coerce to numeric then to Int64 to allow NA before dropping
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values on any required column
    df = df.dropna(subset=required_cols)

    # Convert binary columns to integer type now that NAs are removed
    for c in binary_cols:
        df[c] = df[c].astype(int)

    # Sanity checks / optional clipping: ensure accept and female are 0/1
    df = df[df['accept'].isin([0, 1])]
    df = df[df['female'].isin([0, 1])]

    # Final index reset for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to estimate the effect of gender on mortgage acceptance,
    controlling for observable applicant and loan characteristics. Returns:
    - robust logistic regression results (HC1 standard errors) or a wrapper providing robust stats
    - average marginal effect estimates (overall)

    Model specification (logit):
    accept ~ female + black + housing_expense_ratio + self_employed + married
             + mortgage_credit + consumer_credit + bad_history + PI_ratio
             + loan_to_value + denied_PMI
    """
    df = df.copy()

    # Define predictors (must match columns created/kept by transform)
    predictors = [
        'female',
        'black',
        'housing_expense_ratio',
        'self_employed',
        'married',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'denied_PMI'
    ]

    # Ensure predictors present
    missing = [p for p in predictors + ['accept'] if p not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    X = df[predictors]
    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    try:
        res = logit_model.fit(disp=False)
    except Exception as e:
        # If fit fails (e.g., perfect separation), raise with context
        raise RuntimeError(f"Logit fit failed: {e}")

    # Convert to robust covariance (HC1) for inference.
    # Attempt to use built-in method first; otherwise compute robust covariance manually.
    try:
        res_robust = res.get_robustcov_results(cov_type='HC1')
    except Exception:
        # Compute HC1 robust covariance matrix manually using sandwich estimator.
        # For Logit, score_i = X_i * (y_i - p_i). Meat = sum score_i score_i^T.
        # Robust cov = (H^-1) * Meat * (H^-1) * (n/(n-k))  where H^-1 approximated by res.cov_params()
        # Ensure arrays (not pandas objects) for matrix computations
        X_model = np.asarray(res.model.exog)  # design matrix used in fit as ndarray
        # predicted probabilities
        try:
            p = np.asarray(res.predict(X_model))
        except Exception:
            p = np.asarray(res.predict())
        y_obs = np.asarray(res.model.endog)
        u = (y_obs - p).astype(float)  # residual-like term (score contribution)
        # compute meat: X^T diag(u^2) X efficiently
        S = X_model * u[:, None]
        meat = S.T @ S
        meat = np.asarray(meat)

        # bread: use model-based covariance (inverse Hessian) from result as ndarray
        try:
            bread = np.asarray(res.cov_params())
        except Exception:
            # As a fallback, use pseudo-inverse of X'WX where W = p*(1-p)
            W = p * (1.0 - p)
            XT_W_X = (X_model * W[:, None]).T @ X_model
            bread = np.linalg.pinv(XT_W_X)

        n = int(getattr(res, 'nobs', X_model.shape[0]))
        k = X_model.shape[1]
        # HC1 scaling
        het_scale = n / max(1, (n - k))
        robust_cov = het_scale * bread @ meat @ bread
        robust_cov = np.asarray(robust_cov)

        # Minimal wrapper around original results to expose robust stats
        class RobustResultsWrapper:
            """
            Minimal wrapper around original results to expose:
            - params
            - bse (based on robust covariance)
            - tvalues, pvalues (normal approximation)
            - cov_params() method returning robust covariance
            - delegates other attribute access to the original results object
            """
            def __init__(self, original_res, cov_matrix):
                self._res = original_res
                self._cov = np.asarray(cov_matrix)
                # params from original result
                self.params = np.asarray(original_res.params)
                # robust standard errors
                self.bse = np.sqrt(np.diag(self._cov))
                # t-values and p-values using normal approximation
                # Guard against division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.tvalues = self.params / self.bse
                self.pvalues = 2 * scistats.norm.sf(np.abs(self.tvalues))

            def cov_params(self):
                return self._cov

            def summary(self, *args, **kwargs):
                # Return the original summary (may show original SEs). This wrapper is primarily for access to robust stats.
                return self._res.summary(*args, **kwargs)

            def __getattr__(self, name):
                # Delegate attribute access to the underlying results object when not found here
                return getattr(self._res, name)

        res_robust = RobustResultsWrapper(res, robust_cov)

    # Compute average marginal effects (dy/dx) overall for interpretation
    # Note: marginal effects are computed from the original fitted model object (res)
    try:
        marg = res.get_margeff(at='overall')
        marg_frame = marg.summary_frame()
    except Exception:
        # If marginal effects computation fails, set to None but still return fitted model
        marg_frame = None

    # Return a dictionary containing the robust results object and marginal effects dataframe
    return {
        'logit_result_robust': res_robust,
        'marginal_effects_overall': marg_frame,
        'predictors': predictors
    }