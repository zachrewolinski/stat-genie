from typing import Any
import numpy as np
import pandas as pd
import scipy
from scipy import stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analytic dataframe used for modeling.

    Steps:
    - Make a copy to avoid modifying input.
    - Construct Approved (1 = accepted, 0 = denied) using mortgage decision columns.
      Priority: 'mortgage_credit' described as 1=denied,0=accepted -> Approved = 1 - mortgage_credit.
      If mortgage_credit is not present or not binary, fall back to 'Unnamed: 0' or 'accept'.
    - Construct is_female from 'consumer_credit' (documented as 1 = female, 0 = male). If consumer_credit not present but a 'female' column looks binary, use it.
    - Select a set of controls; coerce to numeric and impute simple median for missing values.
    - Standardize continuous controls (z-score).
    - Ensure all required final columns are present (creating defaults if necessary).
    - Drop rows missing Approved or is_female.

    Returns the transformed dataframe containing the columns used in the model.
    """

    df = df.copy()

    # --- Build Approved outcome robustly ---
    if 'mortgage_credit' in df.columns:
        # According to provided schema: mortgage_credit == 1 means denied, 0 means accepted
        # So Approved (accepted) = 1 - mortgage_credit
        df['Approved'] = (1 - pd.to_numeric(df['mortgage_credit'], errors='coerce')).astype('float')
    elif 'Unnamed: 0' in df.columns:
        df['Approved'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce').astype('float')
    elif 'accept' in df.columns:
        df['Approved'] = pd.to_numeric(df['accept'], errors='coerce').astype('float')
    else:
        raise ValueError("No recognizable mortgage decision column found (expected 'mortgage_credit' or 'Unnamed: 0' or 'accept').")

    # Ensure Approved is binary (0/1). If values are not 0/1, attempt rounding after clipping.
    df['Approved'] = df['Approved'].clip(0, 1)
    df['Approved'] = df['Approved'].round().astype('Int64')

    # --- Build is_female indicator ---
    if 'consumer_credit' in df.columns:
        # Schema notes: consumer_credit == 1 if applicant is female, 0 if male
        df['is_female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').astype('Int64')
    elif 'female' in df.columns:
        # If only 'female' exists and appears binary/near-binary, threshold at 0.5
        fvals = pd.to_numeric(df['female'], errors='coerce')
        df['is_female'] = (fvals > 0.5).astype('Int64')
    else:
        raise ValueError("No recognizable gender column found (expected 'consumer_credit' or 'female').")

    # --- Controls: coerce to numeric and impute medians where necessary ---
    control_cols = [
        'loan_to_value',
        'housing_expense_ratio',
        'denied_PMI',
        'PI_ratio',
        'bad_history',
        'self_employed',
        'married'
    ]

    # Coerce present controls to numeric
    for c in control_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Impute median for numeric controls that exist
    for c in control_cols:
        if c in df.columns:
            if df[c].isna().any():
                median_val = df[c].median()
                df[c] = df[c].fillna(median_val)

    # Standardize continuous-looking controls. We'll treat these as continuous z-scores:
    continuous_candidates = ['loan_to_value', 'housing_expense_ratio', 'denied_PMI', 'PI_ratio']
    for c in continuous_candidates:
        zname = c + '_z'
        if c in df.columns:
            col = df[c].astype(float)
            mean = col.mean()
            std = col.std(ddof=0)
            if std == 0 or np.isnan(std):
                df[zname] = 0.0
            else:
                df[zname] = (col - mean) / std
        else:
            # If original continuous variable not present, create a zero column so final schema is preserved
            df[zname] = 0.0

    # Ensure binary controls are integer 0/1; if missing, create default 0 column
    binary_controls = ['bad_history', 'self_employed', 'married']
    for c in binary_controls:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors='coerce')
            non_na_vals = vals.dropna().unique()
            # Try to coerce to 0/1 where possible
            try:
                unique_set = set(int(x) for x in non_na_vals)
            except Exception:
                unique_set = set()
            if non_na_vals.size == 0:
                df[c] = vals.fillna(0).astype('Int64')
            elif unique_set <= {0, 1}:
                df[c] = vals.astype('Int64')
            else:
                df[c] = (vals > 0.5).astype('Int64')
        else:
            # Create missing binary control with default 0
            df[c] = pd.Series(0, index=df.index).astype('Int64')

    # Final required columns for modeling (exact names required by the contract)
    final_control_cols = [c + '_z' for c in continuous_candidates] + binary_controls

    # Drop rows with missing outcome or missing gender indicator
    df = df.dropna(subset=['Approved', 'is_female'])

    # Cast final binary/integer columns to numeric types expected by model
    df['Approved'] = df['Approved'].astype(int)
    df['is_female'] = df['is_female'].astype(int)
    for c in binary_controls:
        # ensure binary control dtype int
        df[c] = df[c].astype(int)

    # Keep only the columns needed for modeling to make downstream code simpler
    keep_cols = ['Approved', 'is_female'] + final_control_cols
    # Ensure all keep_cols exist (they should, by construction above)
    missing_keep = [c for c in keep_cols if c not in df.columns]
    if missing_keep:
        # As a safeguard, create any missing columns with zeros (should not normally happen)
        for c in missing_keep:
            if c.endswith('_z'):
                df[c] = 0.0
            else:
                df[c] = 0
    df_model = df[keep_cols].copy()

    return df_model


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting Approved from is_female and controls.

    Uses statsmodels Logit and returns results augmented with HC1 robust covariance
    information. If the Statsmodels results object does not support the convenience
    wrapper for robust covariances in this environment, compute robust covariance
    manually (HC1) and return a simple results-like object with key attributes:
      params, bse (robust), pvalues (robust), conf_int (robust), cov_robust
    and a summary() method for printing.
    """

    # Ensure required columns present
    required = ['Approved', 'is_female']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Build covariates list: must use only the specified conceptual controls
    control_cols_z = ['loan_to_value_z', 'housing_expense_ratio_z', 'denied_PMI_z', 'PI_ratio_z']
    binary_controls = ['bad_history', 'self_employed', 'married']
    covariates = ['is_female'] + control_cols_z + binary_controls

    # Ensure covariates exist in dataframe
    missing_cov = [c for c in covariates if c not in df.columns]
    if missing_cov:
        raise ValueError(f"Model dataframe is missing required control columns: {missing_cov}")

    X = df[covariates].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['Approved'].astype(float)

    # Fit logistic regression
    logit = sm.Logit(y, X)
    try:
        res = logit.fit(disp=False)
    except Exception:
        # Try a more robust solver if default fails
        res = logit.fit(disp=False, method='bfgs', maxiter=200)

    # Try to obtain robust covariance via results convenience method if available
    try:
        # Some environments may provide get_robustcov_results; try it first
        robust_wrapper = res.get_robustcov_results(cov_type='HC1')  # type: ignore[attr-defined]
        print(robust_wrapper.summary())
        return robust_wrapper
    except Exception:
        # Fallback: compute HC1 robust covariance matrix manually using sandwich formula
        # robust_cov = inv(H) * (X' diag(resid^2) X) * inv(H) scaled by n/(n-k)
        # where inv(H) is the model-based covariance (res.cov_params())
        # Residuals: prefer resid_response, else compute y - mu
        if hasattr(res, 'resid_response'):
            resid = np.asarray(res.resid_response)
        else:
            try:
                resid = np.asarray(y - res.predict(X))
            except Exception:
                # As a last resort try res.resid
                if hasattr(res, 'resid'):
                    resid = np.asarray(res.resid)
                else:
                    raise RuntimeError("Unable to obtain residuals for robust covariance computation.")

        exog = np.asarray(res.model.exog)
        n, k = exog.shape

        # Compute meat = X' diag(resid^2) X
        # Use resid squared
        resid_sq = resid ** 2
        # Efficient multiplication without forming full diag matrix:
        # compute (X * resid[:, None]).T @ (X * resid[:, None]) equals X.T @ diag(resid^2) @ X
        xr = exog * resid_sq[:, None]
        meat = exog.T @ xr

        # inv_hess: use model-based covariance as an estimate of inv(-H)
        try:
            inv_hess = np.asarray(res.cov_params())
        except Exception:
            # If cov_params cannot be obtained, fall back to pseudo-inverse of the Hessian
            try:
                hess = -res.model.hessian(res.params) if hasattr(res.model, 'hessian') else res.hessian()
                inv_hess = np.linalg.pinv(hess)
            except Exception:
                # Last resort: use pseudo-inverse of X'X
                inv_hess = np.linalg.pinv(exog.T @ exog)

        robust_cov = inv_hess @ meat @ inv_hess

        # HC1 scaling: multiply by n/(n - k)
        if n - k > 0:
            scale = float(n) / float(n - k)
            robust_cov = robust_cov * scale

    # Compute robust standard errors, z-stats, p-values and confidence intervals
    params = res.params
    param_index = params.index if hasattr(params, 'index') else None
    bse = pd.Series(np.sqrt(np.abs(np.diag(robust_cov))), index=param_index)
    z_values = params / bse
    pvalues = 2 * (1 - sps.norm.cdf(np.abs(z_values)))
    conf_int_lower = params - 1.96 * bse
    conf_int_upper = params + 1.96 * bse
    conf_int = pd.DataFrame({
        '2.5%': conf_int_lower,
        '97.5%': conf_int_upper
    }, index=param_index)

    # Prepare a simple results-like object to return
    class RobustResult:
        def __init__(self, params, bse, pvalues, conf_int, cov_robust, orig_res):
            self.params = params
            self.bse = bse
            self.pvalues = pvalues
            self.conf_int = conf_int
            self.cov_robust = cov_robust
            self.model_result = orig_res

        def summary(self):
            tbl = pd.DataFrame({
                'coef': self.params,
                'std err': self.bse,
                'z': self.params / self.bse,
                'P>|z|': self.pvalues,
                '2.5%': self.conf_int['2.5%'],
                '97.5%': self.conf_int['97.5%'],
            })
            summary_str = "\nRobust (HC1) logistic regression results\n"
            summary_str += tbl.to_string(float_format=lambda x: f"{x:0.4f}")
            print(summary_str)
            return summary_str

        # allow attribute access similar to statsmodels results
        def __getattr__(self, item):
            return getattr(self.model_result, item)

    robust_res = RobustResult(params=params, bse=bse, pvalues=pvalues, conf_int=conf_int, cov_robust=robust_cov, orig_res=res)

    # Print a concise summary and return the robust results object
    robust_res.summary()
    return robust_res