from typing import Any
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
from statsmodels.stats.sandwich_covariance import cov_hc3


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/negative_leading_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston mortgage dataset to the analysis-ready dataframe.

    Produces standardized continuous controls and drops rows with missing values in any
    variables that will be used in the model.

    Final dataframe includes the exact columns used in modeling:
      - accept (DV), female (IV)
      - control columns: black, housing_expense_ratio_z, self_employed, married,
        mortgage_credit_z, consumer_credit_z, bad_history, PI_ratio_z, loan_to_value_z, denied_PMI
    """
    df = df.copy()

    # Ensure essential columns are present
    required_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Keep only relevant columns first to simplify
    df = df[required_cols].copy()

    # Coerce types
    # Binary indicators should be numeric 0/1
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Continuous controls -> coerce to numeric
    cont_cols = ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit']
    for c in cont_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any variables used by the model
    df = df.dropna(subset=required_cols)

    # Standardize continuous controls (z-score). Keep original columns removed from controls
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If zero variance, create zeros
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # The final dataframe should contain the columns referenced in the modeling code
    final_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio_z', 'self_employed', 'married',
        'mortgage_credit_z', 'consumer_credit_z', 'bad_history', 'PI_ratio_z', 'loan_to_value_z', 'denied_PMI'
    ]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting mortgage acceptance (accept) from applicant gender (female)
    controlling for applicant and loan characteristics.

    Returns a dictionary with:
      - model: the fitted robust-logit results object (statsmodels-like wrapper)
      - summary_text: textual model summary
      - female_odds_ratio, female_pvalue, female_ci_lower, female_ci_upper
      - marginal_effects_summary: textual summary of average marginal effects
    """
    # Copy to avoid side-effects
    df = df.copy()

    # Ensure required columns exist
    model_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio_z', 'self_employed', 'married',
        'mortgage_credit_z', 'consumer_credit_z', 'bad_history', 'PI_ratio_z', 'loan_to_value_z', 'denied_PMI'
    ]
    missing = [c for c in model_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Define X and y
    y = df['accept'].astype(float)
    X = df[[
        'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'housing_expense_ratio_z', 'PI_ratio_z', 'loan_to_value_z', 'mortgage_credit_z', 'consumer_credit_z'
    ]].astype(float)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood). We'll request the fit and then compute robust SEs.
    model_logit = sm.Logit(y, X)
    try:
        res = model_logit.fit(disp=False, method='lbfgs', maxiter=100)
    except Exception:
        # fallback to default method
        res = model_logit.fit(disp=False)

    # Attempt to obtain a HC3 robust covariance using statsmodels helper. If unavailable, fall back.
    cov = None
    try:
        # Preferred: ask the fitted results to return a robust-covariance results object
        res_sm_robust = res.get_robustcov_results(cov_type='HC3')
        cov = res_sm_robust.cov_params()
    except Exception:
        try:
            # Fallback: use cov_hc3 directly on the results object (may fail for some discrete models)
            cov = cov_hc3(res)
        except Exception:
            # Last resort: use model-based covariance (not robust)
            cov = res.cov_params()

    # Ensure cov is an ndarray or DataFrame; extract diagonal safely
    if isinstance(cov, pd.DataFrame):
        cov_values = cov.values
    else:
        cov_values = np.asarray(cov)

    params = res.params.copy()
    bse_array = np.sqrt(np.diag(cov_values))
    bse = pd.Series(bse_array, index=params.index)
    tvals = params / bse
    # Ensure pvalues is a Series with the same index (previously was converted to ndarray)
    pvalues = pd.Series(2 * (1 - scipy.stats.norm.cdf(np.abs(tvals.values))), index=params.index)

    def conf_int(alpha=0.05):
        z = scipy.stats.norm.ppf(1 - alpha / 2)
        lower = params - z * bse
        upper = params + z * bse
        return pd.DataFrame({0: lower, 1: upper})

    # Create a simple wrapper object that mimics the subset of a results object used downstream
    class RobustResultsWrapper:
        def __init__(self, params, bse, pvalues, cov, conf_func, orig_res):
            self.params = params
            self.bse = bse
            self.pvalues = pvalues
            self._cov = cov
            self._conf_func = conf_func
            self._orig_res = orig_res

        def conf_int(self, alpha=0.05):
            return self._conf_func(alpha=alpha)

        def summary(self):
            # Build a textual summary that mirrors the typical table with robust SEs
            tab = pd.DataFrame({
                'coef': self.params,
                'std err': self.bse,
                'z': (self.params / self.bse),
                'P>|z|': self.pvalues,
            })
            ci = self.conf_int()
            tab['[0.025'] = ci[0]
            tab['0.975]'] = ci[1]
            return tab

        def summary_as_text(self):
            return self.summary().to_string()

        def cov_params(self):
            # Return covariance as a DataFrame with appropriate index/columns
            cov_arr = np.asarray(self._cov)
            return pd.DataFrame(cov_arr, index=self.params.index, columns=self.params.index)

    res_robust = RobustResultsWrapper(params=params, bse=bse, pvalues=pvalues, cov=cov_values, conf_func=conf_int, orig_res=res)

    # Odds ratio and confidence interval for female (using robust parameters/cov).
    if 'female' not in res_robust.params.index:
        raise KeyError('female not in fitted model parameters')

    female_coef = float(res_robust.params['female'])
    female_pval = float(res_robust.pvalues['female'])
    female_ci = res_robust.conf_int().loc['female'].astype(float)
    female_odds = float(np.exp(female_coef))
    female_ci_lower = float(np.exp(female_ci[0]))
    female_ci_upper = float(np.exp(female_ci[1]))

    # Average marginal effect for female (change in probability)
    try:
        me = res.get_margeff(method='dydx', at='overall')
        me_summary = me.summary().as_text()
        # attempt to access table HTML if needed; keep within try/except
        try:
            me_table = me.summary().tables[1].as_html()
        except Exception:
            me_table = None
    except Exception:
        me = None
        me_summary = 'marginal effects could not be computed'
        me_table = None

    # Prepare textual summary_text using the robust summary table
    summary_text = res_robust.summary_as_text()

    results = {
        'model_object': res_robust,
        'summary_text': summary_text,
        'female_odds_ratio': female_odds,
        'female_pvalue': female_pval,
        'female_ci_lower': female_ci_lower,
        'female_ci_upper': female_ci_upper,
        'marginal_effects_summary': me_summary
    }

    return results