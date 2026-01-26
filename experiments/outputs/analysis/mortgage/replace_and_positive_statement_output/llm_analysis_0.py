from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Note: This top-level read is kept from the original file but is not required
# for the transform/model functions to operate. The functions operate on any
# dataframe passed to them.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_and_positive_statement_output/mortgage.csv')
except Exception:
    # If the file is not present in the environment where this module is imported,
    # avoid raising an import-time error. The transform/model functions can still be used.
    df = None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston Fed mortgage dataset to produce the final dataframe used in modeling.

    Steps:
    - Create binary dependent variable 'accepted' from 'accept'.
    - Ensure 'female' is numeric binary.
    - Drop rows with missing values in outcome, IV, and required controls.
    - Standardize continuous/ordinal numeric predictors to make coefficients comparable and numerically stable.
    - Return a dataframe containing only the columns used in the regression model.
    """
    df = df.copy()

    # Ensure expected columns exist
    required_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Dependent variable: accepted (1) / denied (0)
    df['accepted'] = df['accept'].astype(float)

    # Independent variable: female (ensure binary numeric)
    df['female'] = df['female'].astype(float)

    # Drop rows with missing values on the variables we will use
    drop_on = required_cols + ['accept', 'female']
    df = df.dropna(subset=drop_on)

    # Cast binary indicators to ints
    bin_cols = ['black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in bin_cols:
        df[c] = df[c].astype(int)

    # Standardize continuous and ordinal predictors (z-score). Use population std (ddof=0) for consistency.
    # Map source column -> target standardized column name (must match required final column names)
    z_map = {
        'housing_expense_ratio': 'z_housing_exp_ratio',
        'PI_ratio': 'z_PI_ratio',
        'loan_to_value': 'z_loan_to_value',
        'mortgage_credit': 'z_mortgage_credit',
        'consumer_credit': 'z_consumer_credit'
    }
    for src, tgt in z_map.items():
        mean = df[src].mean()
        std = df[src].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, create zero column to avoid division by zero
            df[tgt] = 0.0
        else:
            df[tgt] = (df[src] - mean) / std

    # Final columns to return (only what the model will need)
    final_cols = [
        'accepted', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'z_housing_exp_ratio', 'z_PI_ratio', 'z_loan_to_value', 'z_mortgage_credit', 'z_consumer_credit'
    ]

    return df[final_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to estimate the effect of gender (female) on loan acceptance,
    controlling for applicant characteristics. Returns the fitted model object, text summary,
    and a table of odds ratios with 95% confidence intervals.

    Model specification:
      accepted ~ female + black + self_employed + married + bad_history + denied_PMI
                 + z_housing_exp_ratio + z_PI_ratio + z_loan_to_value + z_mortgage_credit + z_consumer_credit

    Uses robust (HC1) standard errors when available; otherwise computes HC1 robust cov manually.
    """
    # Required predictor columns
    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'z_housing_exp_ratio', 'z_PI_ratio', 'z_loan_to_value', 'z_mortgage_credit', 'z_consumer_credit'
    ]

    missing = [c for c in ['accepted'] + X_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare X and y
    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['accepted'].astype(int)

    # Fit logistic regression
    logit_res = sm.Logit(y, X).fit(disp=False)

    # Attempt to obtain robust-covariance wrapped results if available; otherwise compute HC1 robust cov manually.
    try:
        # Some versions of statsmodels provide this method
        robust_res = logit_res.get_robustcov_results(cov_type='HC1')
        params = robust_res.params
        conf = robust_res.conf_int()
        summary_text = robust_res.summary2().as_text()
    except Exception:
        # Fallback: compute HC1 sandwich covariance matrix and derive robust SEs and CIs
        # Compute residuals as response residuals (y - mu). Use predict on the fitted model.
        try:
            pred = logit_res.predict(X)
        except Exception:
            # As a final fallback, use model's predict with params
            pred = logit_res.model.predict(logit_res.params)

        resid = (y - pred).to_numpy() if isinstance(y, pd.Series) else (y - pred)
        # nobs and df_resid: prefer attributes on results if present, else compute
        nobs = getattr(logit_res, 'nobs', X.shape[0])
        df_resid = getattr(logit_res, 'df_resid', None)
        if df_resid is None:
            # approximate df_resid as nobs - number of parameters
            df_resid = int(nobs) - X.shape[1]

        # HC1 scaling
        het_scale = (nobs / df_resid) * (resid ** 2)

        # Build arrays for sandwich computation
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        # Compute inverse of X'X
        XTX = np.dot(X_arr.T, X_arr)
        try:
            XTXinv = np.linalg.inv(XTX)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            XTXinv = np.linalg.pinv(XTX)

        # Compute middle term: X' * diag(het_scale) * X
        # First multiply rows of X by het_scale
        # (X_arr.T * het_scale) has shape (k, n), dot with X_arr (n, k) => (k, k)
        S = np.dot((X_arr.T * het_scale), X_arr)
        cov_robust = XTXinv.dot(S).dot(XTXinv)

        params = logit_res.params
        se = np.sqrt(np.diag(cov_robust))
        z = 1.96
        conf = pd.DataFrame({
            '2.5%': params - z * se,
            '97.5%': params + z * se
        }, index=params.index)
        summary_text = logit_res.summary2().as_text()

        # Create a lightweight object to mimic robust result behavior for downstream usage
        class _RobustFallback:
            def __init__(self, orig_res, covmatrix, conf_df):
                self.orig_res = orig_res
                self.cov = covmatrix
                self.params = orig_res.params
                self._conf = conf_df

            def conf_int(self):
                return self._conf

            def summary2(self):
                return self.orig_res.summary2()

        robust_res = _RobustFallback(logit_res, cov_robust, conf)

    # Ensure conf has expected column names
    if isinstance(conf, pd.DataFrame):
        if conf.shape[1] == 2:
            conf = conf.copy()
            conf.columns = ['2.5%', '97.5%']
    else:
        # As a fallback, construct conf from params with +/- 1.96*se if possible
        conf = pd.DataFrame(index=params.index, columns=['2.5%', '97.5%'])
        conf['2.5%'] = np.nan
        conf['97.5%'] = np.nan

    # Compute odds ratios and CIs
    odds = np.exp(params)
    ci_lower = np.exp(conf['2.5%'])
    ci_upper = np.exp(conf['97.5%'])

    or_table = pd.DataFrame({
        'OR': odds,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'coef': params,
    })

    # Pack results
    results = {
        'model_object': logit_res,
        'robust_results': robust_res,
        'summary_text': summary_text,
        'odds_ratios': or_table
    }

    return results