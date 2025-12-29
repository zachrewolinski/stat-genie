from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/affairs/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid modifying original
    df = df.copy()

    # Keep only rows that have non-missing values for the variables needed in the model
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_cols)

    # Ensure 'affairs' numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Normalize and encode 'children' to binary: 1 = yes, 0 = no
    # Handle capitalization and stray whitespace
    df['children_clean'] = df['children'].astype(str).str.strip().str.lower()
    df['children_binary'] = df['children_clean'].map({'yes': 1, 'no': 0})

    # Normalize and encode gender to male indicator: 1 = male, 0 = female
    df['gender_clean'] = df['gender'].astype(str).str.strip().str.lower()
    df['gender_male'] = df['gender_clean'].map({'male': 1, 'female': 0})

    # If mapping produced NaNs (unexpected categories), drop those rows
    df = df.dropna(subset=['children_binary', 'gender_male'])

    # Ensure numeric controls are numeric
    num_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop any remaining rows with NA in numeric controls or affairs
    df = df.dropna(subset=['affairs'] + num_cols)

    # Create interaction term (children x gender) for moderation test
    df['children_gender_interaction'] = df['children_binary'] * df['gender_male']

    # Keep only columns needed for modeling (but returning full df is acceptable). We'll ensure model columns exist.
    model_cols = ['affairs', 'children_binary', 'gender_male', 'children_gender_interaction'] + num_cols
    # Return df with all columns but guarantee model columns exist
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import numpy as np
    import statsmodels.api as sm
    from scipy.stats import norm
    from statsmodels.base.model import GenericLikelihoodModel

    # Build model matrix
    X_cols = ['const', 'children_binary', 'gender_male', 'children_gender_interaction',
              'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    df = df.copy()
    df['const'] = 1.0
    X = df[X_cols].astype(float)
    y = df['affairs'].astype(float)

    # Define Tobit (censored regression) using GenericLikelihoodModel
    class TobitLL(GenericLikelihoodModel):
        def __init__(self, endog, exog, left=0.0, right=12.0, **kwds):
            self.left = left
            self.right = right
            super(TobitLL, self).__init__(endog, exog, **kwds)

        def loglike(self, params):
            # params: [beta_0, beta_1, ..., beta_k, sigma]
            k = self.exog.shape[1]
            beta = params[:k]
            sigma = params[k]
            if sigma <= 0 or np.isnan(sigma):
                return -1e12
            mu = np.dot(self.exog, beta)
            y = self.endog
            left = self.left
            right = self.right

            # masks
            is_left = (y <= left + 1e-8)
            is_right = (y >= right - 1e-8)
            is_uncensored = ~(is_left | is_right)

            ll = np.zeros_like(y, dtype=float)

            # left-censored observations: log Phi( (left - mu) / sigma )
            if is_left.any():
                z_left = (left - mu[is_left]) / sigma
                ll[is_left] = norm.logcdf(z_left)

            # right-censored observations: log (1 - Phi( (right - mu)/sigma ))
            if is_right.any():
                z_right = (right - mu[is_right]) / sigma
                # 1 - cdf may underflow; use logsf
                ll[is_right] = norm.logsf(z_right)

            # uncensored: normal logpdf
            if is_uncensored.any():
                z = (y[is_uncensored] - mu[is_uncensored]) / sigma
                ll[is_uncensored] = norm.logpdf(z) - np.log(sigma)

            return np.sum(ll)

    # Prepare the model: exog matrix and endog vector
    exog = X.values
    endog = y.values

    # Fit starting values with OLS for betas and residual sd for sigma
    ols_res = sm.OLS(endog, exog).fit()
    start_beta = ols_res.params
    resid = endog - np.dot(exog, start_beta)
    start_sigma = np.std(resid)
    if start_sigma <= 0 or np.isnan(start_sigma):
        start_sigma = 1.0

    start_params = np.r_[start_beta, start_sigma]

    # Instantiate Tobit model and fit
    tob = TobitLL(endog, exog, left=0.0, right=12.0)
    # Use method='bfgs' (or 'nm') and let it estimate
    try:
        res = tob.fit(start_params=start_params, method='bfgs', disp=False)
    except Exception:
        # fallback to Nelder-Mead if BFGS has trouble
        res = tob.fit(start_params=start_params, method='nm', disp=False)

    # For convenience, attach the names to params in the result summary
    param_names = X_cols + ['sigma']
    # Create a small summary dictionary to return along with the raw results
    summary = {
        'params': pd.Series(res.params, index=param_names),
        'bse': pd.Series(res.bse, index=param_names) if hasattr(res, 'bse') and res.bse is not None else None,
        'llf': getattr(res, 'llf', None),
        'converged': (res.mle_retvals.get('converged', True) if hasattr(res, 'mle_retvals') and isinstance(res.mle_retvals, dict) else True),
        'nobs': int(getattr(res, 'nobs', len(endog)))
    }

    # Return the raw result object and the convenient summary dict
    return {'result_obj': res, 'summary': summary}