from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_hc3

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston Fed mortgage dataset for logistic regression.

    Steps performed:
    - Make a copy of the dataframe
    - Ensure the key binary columns are numeric and in {0,1}
    - Convert continuous predictors to numeric and create standardized (z-scored) versions for modeling
    - Drop rows with missing values in any variable used by the model
    - Return the dataframe containing the original + derived columns required for the model
    """
    df = df.copy()

    # Ensure target and key binaries are numeric
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure continuous predictors are numeric
    cont_cols = ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit']
    for c in cont_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create standardized versions (z-scores) for continuous controls used in the model
    # Use sample std (pandas default ddof=1)
    if 'housing_expense_ratio' in df.columns:
        df['housing_expense_ratio_std'] = (df['housing_expense_ratio'] - df['housing_expense_ratio'].mean()) / df['housing_expense_ratio'].std()
    else:
        df['housing_expense_ratio_std'] = np.nan

    if 'PI_ratio' in df.columns:
        df['PI_ratio_std'] = (df['PI_ratio'] - df['PI_ratio'].mean()) / df['PI_ratio'].std()
    else:
        df['PI_ratio_std'] = np.nan

    if 'loan_to_value' in df.columns:
        df['loan_to_value_std'] = (df['loan_to_value'] - df['loan_to_value'].mean()) / df['loan_to_value'].std()
    else:
        df['loan_to_value_std'] = np.nan

    if 'mortgage_credit' in df.columns:
        df['mortgage_credit_std'] = (df['mortgage_credit'] - df['mortgage_credit'].mean()) / df['mortgage_credit'].std()
    else:
        df['mortgage_credit_std'] = np.nan

    if 'consumer_credit' in df.columns:
        df['consumer_credit_std'] = (df['consumer_credit'] - df['consumer_credit'].mean()) / df['consumer_credit'].std()
    else:
        df['consumer_credit_std'] = np.nan

    # List of columns required by the model
    required_cols = [
        'accept',
        'female',
        'black',
        'housing_expense_ratio_std',
        'self_employed',
        'married',
        'mortgage_credit_std',
        'consumer_credit_std',
        'bad_history',
        'PI_ratio_std',
        'loan_to_value_std',
        'denied_PMI'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Cast binary variables to integer type (0/1)
    for c in ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']:
        if c in df.columns:
            # Values are already numeric and rows with missing values dropped; cast to integer
            df[c] = df[c].astype(int)

    # Return the full dataframe (with derived standardized columns available).
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to estimate the effect of applicant gender on mortgage acceptance,
    controlling for applicant financial and demographic covariates.

    Model specification:
    logit( P(accept=1) ) = alpha + beta_female * female + sum(gamma_k * control_k)

    Returns a dictionary with the robust fitted model object and a summary table of odds ratios,
    confidence intervals and p-values.
    """
    # Columns used in the model (must match those created in transform)
    model_cols = [
        'female',
        'black',
        'housing_expense_ratio_std',
        'self_employed',
        'married',
        'mortgage_credit_std',
        'consumer_credit_std',
        'bad_history',
        'PI_ratio_std',
        'loan_to_value_std',
        'denied_PMI'
    ]

    # Ensure the dataframe contains the needed columns
    missing = [c for c in model_cols + ['accept'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    y = df['accept']
    X = df[model_cols]

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit_mod = sm.Logit(y, X)
    logit_res = logit_mod.fit(disp=False)

    # Some statsmodels sandwich covariance helpers expect a 'resid' attribute on results.
    # LogitResults does not always expose 'resid', so attach a response residuals array if missing.
    # Use response residuals: observed - predicted probability
    if not hasattr(logit_res, 'resid'):
        try:
            resid_arr = logit_res.model.endog - logit_res.predict()
        except Exception:
            # Fallback: try resid_response if present
            resid_arr = getattr(logit_res, 'resid_response', None)
            if resid_arr is None:
                raise
        setattr(logit_res, 'resid', resid_arr)

    # Some versions of statsmodels' cov_hc3 expect the model to have a pinv_wexog attribute.
    # Ensure that attribute exists by computing a pseudo-inverse of the design matrix if necessary.
    if not hasattr(logit_res.model, 'pinv_wexog'):
        try:
            exog = getattr(logit_res.model, 'exog', None)
            if exog is not None:
                pinv = np.linalg.pinv(exog)
                setattr(logit_res.model, 'pinv_wexog', pinv)
        except Exception:
            # If we cannot set pinv_wexog, allow cov_hc3 to raise its own error.
            pass

    # Compute robust (HC3) covariance matrix using statsmodels' sandwich covariance utility
    robust_cov_arr = cov_hc3(logit_res)
    params = logit_res.params
    # Convert to DataFrame for labeling
    robust_cov = pd.DataFrame(robust_cov_arr, index=params.index, columns=params.index)

    # Robust standard errors as a Series aligned with params index
    robust_se_series = pd.Series(np.sqrt(np.diag(robust_cov)), index=params.index)

    # z-scores using robust standard errors
    z_scores = params / robust_se_series
    pvalues = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # 95% robust confidence intervals on parameter scale
    z_975 = stats.norm.ppf(0.975)
    conf_lower = params - z_975 * robust_se_series
    conf_upper = params + z_975 * robust_se_series

    # Compute odds ratios and robust confidence intervals for OR
    or_vals = np.exp(params)
    conf_or_lower = np.exp(conf_lower)
    conf_or_upper = np.exp(conf_upper)

    summary_table = pd.DataFrame({
        'OR': or_vals,
        '2.5%': conf_or_lower,
        '97.5%': conf_or_upper,
        'pvalue': pvalues
    }, index=params.index)

    # Sort so key variable female is near the top for readability; put const last
    idx = list(summary_table.index)
    ordered_index = []
    if 'female' in idx:
        ordered_index.append('female')
    for c in idx:
        if c not in ('female', 'const'):
            ordered_index.append(c)
    if 'const' in idx:
        ordered_index.append('const')
    summary_table = summary_table.loc[ordered_index]

    # Create a small wrapper object to hold robust results-like attributes
    class RobustResults:
        def __init__(self, orig_res, cov_rob_df, bse_series, pvals_series):
            self.orig_res = orig_res
            self.params = orig_res.params
            self._cov = cov_rob_df
            # cov_params method to mimic statsmodels results
            self.cov_params = lambda: self._cov
            # bse as Series aligned with params for convenience
            self.bse = bse_series
            self.pvalues = pvals_series

        def conf_int(self, alpha=0.05):
            z = stats.norm.ppf(1 - alpha / 2)
            lower = self.params - z * self.bse
            upper = self.params + z * self.bse
            ci = pd.DataFrame({0: lower, 1: upper}, index=self.params.index)
            return ci

    robust_res = RobustResults(logit_res, robust_cov, robust_se_series, pvalues)

    return {
        'model_results_robust': robust_res,
        'summary_table': summary_table
    }