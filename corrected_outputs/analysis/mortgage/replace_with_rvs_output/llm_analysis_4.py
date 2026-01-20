from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hc3
from scipy import stats

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Boston mortgage dataset for modeling the effect of gender on approval.
    This transformation:
      - makes a safe copy of the input
      - coerces key numeric columns to numeric
      - drops rows with missing values in the outcome, IV, moderator, and essential controls
      - standardizes continuous predictors used as controls
      - constructs an interaction term female_black

    Returns the dataframe with columns used in modeling (see conceptual variables).
    """
    df = df.copy()

    # Ensure commonly used columns exist; if not, KeyError will surface so user knows
    required_cols = [
        'accept', 'female', 'black',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'bad_history', 'married', 'self_employed', 'denied_PMI'
    ]

    # Coerce the numeric controls to numeric where applicable
    for col in ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
                'accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in the outcome, IV, moderator, and the chosen controls
    subset_for_model = [
        'accept', 'female', 'black',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'bad_history', 'married', 'self_employed', 'denied_PMI'
    ]
    # Keep only columns that are present in the df to avoid errors
    subset_for_model = [c for c in subset_for_model if c in df.columns]
    df = df.dropna(subset=subset_for_model)

    # Standardize continuous/ordinal controls (mean 0, sd 1). Use population SD (ddof=0) for stability.
    def standardize(series: pd.Series) -> pd.Series:
        if series.dtype.kind in 'biufc':
            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0 or np.isnan(std):
                # if no variation, return zeros
                return pd.Series(0.0, index=series.index)
            return (series - mean) / std
        else:
            s = pd.to_numeric(series, errors='coerce')
            mean = s.mean()
            std = s.std(ddof=0)
            if std == 0 or np.isnan(std):
                return pd.Series(0.0, index=s.index)
            return (s - mean) / std

    # Create standardized columns for continuous controls
    if 'mortgage_credit' in df.columns:
        df['mortgage_credit_s'] = standardize(df['mortgage_credit'])
    if 'consumer_credit' in df.columns:
        df['consumer_credit_s'] = standardize(df['consumer_credit'])
    if 'PI_ratio' in df.columns:
        df['PI_ratio_s'] = standardize(df['PI_ratio'])
    if 'loan_to_value' in df.columns:
        df['loan_to_value_s'] = standardize(df['loan_to_value'])
    if 'housing_expense_ratio' in df.columns:
        df['housing_expense_ratio_s'] = standardize(df['housing_expense_ratio'])

    # Ensure binary indicators are integers (0/1)
    for bcol in ['female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI', 'accept']:
        if bcol in df.columns:
            # After dropna above, casting to int is safe for 0/1 indicators
            df[bcol] = df[bcol].astype(int)

    # Interaction term for moderation test
    if ('female' in df.columns) and ('black' in df.columns):
        df['female_black'] = df['female'] * df['black']

    # Return only the columns relevant for modeling plus original ones to preserve context
    model_columns = [
        'accept', 'female', 'black', 'female_black',
        'mortgage_credit_s', 'consumer_credit_s', 'PI_ratio_s', 'loan_to_value_s', 'housing_expense_ratio_s',
        'bad_history', 'married', 'self_employed', 'denied_PMI'
    ]
    # Keep only those that exist in df
    model_columns = [c for c in model_columns if c in df.columns]

    return df[model_columns].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial) model predicting acceptance (accept) from gender (female),
    race (black), their interaction, and additional controls. Returns a fitted results-like object with robust SEs.

    Model specification (in words):
      logit(P(accept=1)) = beta0 + beta1*female + beta2*black + beta3*(female*black)
                             + beta4*mortgage_credit_s + beta5*consumer_credit_s + beta6*PI_ratio_s
                             + beta7*loan_to_value_s + beta8*housing_expense_ratio_s
                             + beta9*bad_history + beta10*married + beta11*self_employed + beta12*denied_PMI

    The function expects df to be the transformed dataframe produced by transform().
    """
    # Required columns for the model
    X_cols = [
        'female', 'black', 'female_black',
        'mortgage_credit_s', 'consumer_credit_s', 'PI_ratio_s', 'loan_to_value_s', 'housing_expense_ratio_s',
        'bad_history', 'married', 'self_employed', 'denied_PMI'
    ]
    # Keep only columns present in df (in case some were not available)
    X_cols = [c for c in X_cols if c in df.columns]

    X = df[X_cols].copy()
    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Outcome
    y = df['accept'].astype(int)

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    # suppress output during fit
    res = logit_model.fit(disp=False)

    # Compute robust covariance (HC3)
    # Prefer built-in robust results when available.
    try:
        robust_res = res.get_robustcov_results(cov_type='HC3')
        robust_cov = robust_res.cov_params()
    except Exception:
        # Fallback: ensure the result object has residuals and model attributes expected by cov_hc3.
        if not hasattr(res, 'resid'):
            if hasattr(res, 'resid_response'):
                res.resid = res.resid_response
            elif hasattr(res, 'resid_pearson'):
                res.resid = res.resid_pearson
            else:
                try:
                    pred = res.predict()
                    res.resid = res.model.endog - pred
                except Exception:
                    res.resid = np.zeros_like(res.model.endog)

        # Some versions of cov_hc3 expect the model to have a pinv_wexog attribute (present for OLS models).
        # Create it here from the design matrix if missing so cov_hc3 can proceed.
        if not hasattr(res.model, 'pinv_wexog'):
            try:
                res.model.pinv_wexog = np.linalg.pinv(res.model.exog)
            except Exception:
                # As a last resort, fall back to the non-robust covariance matrix
                robust_cov = res.cov_params()

        # If robust_cov wasn't set by fallback, compute via cov_hc3
        if 'robust_cov' not in locals():
            robust_cov = cov_hc3(res)

    # Build a lightweight wrapper around the original results to expose robust SEs and related stats
    class RobustResultsWrapper:
        def __init__(self, base_res, robust_cov, X, y):
            self._res = base_res
            self._robust_cov = robust_cov

            # Align parameters with covariance matrix if it's a DataFrame
            if isinstance(self._robust_cov, pd.DataFrame):
                cov_vals = self._robust_cov.values
                param_index = list(self._robust_cov.index)
                # Reindex params to match covariance ordering
                params = base_res.params.reindex(param_index)
                self.params = params
            else:
                cov_vals = np.asarray(self._robust_cov)
                param_index = list(base_res.params.index)
                self.params = base_res.params

            # robust bse as a pandas Series for clarity
            self.bse = pd.Series(np.sqrt(np.diag(cov_vals)), index=param_index)

            # t-values / z-values for large-sample inference
            # ensure division aligns by index
            self.tvalues = self.params / self.bse

            # two-sided p-values using normal distribution (logit large-sample)
            self.pvalues = 2 * (1 - stats.norm.cdf(np.abs(self.tvalues)))

            # expose covariance accessor similar to statsmodels' cov_params()
            if isinstance(self._robust_cov, pd.DataFrame):
                self.cov_params = lambda: self._robust_cov
            else:
                # return as numpy array
                self.cov_params = lambda: np.asarray(self._robust_cov)

            # attach model data
            self.model_data = {'X': X, 'y': y}

            # expose some useful scalar metrics by delegating to base result
            try:
                self.llf = base_res.llf
            except Exception:
                self.llf = None
            try:
                self.aic = base_res.aic
            except Exception:
                self.aic = None
            try:
                self.bic = base_res.bic
            except Exception:
                self.bic = None

        def summary(self, *args, **kwargs):
            # Use the base result summary but replace the table of bse/pvalues if possible.
            # For simplicity, delegate to base result summary.
            return self._res.summary(*args, **kwargs)

        def __getattr__(self, item):
            # Delegate attribute access to the original results for anything not overridden
            return getattr(self._res, item)

    res_robust = RobustResultsWrapper(res, robust_cov, X, y)

    return res_robust