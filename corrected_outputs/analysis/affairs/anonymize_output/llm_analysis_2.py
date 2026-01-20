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
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/anonymize_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original Fair (Psychology Today) dataset into a cleaned dataframe
    with the exact column names used in the model.

    Output columns (kept/created):
      - Affairs: numeric, original feature2 (frequency of extramarital sex)
      - Children: binary (1=yes, 0=no) from feature6
      - Female: binary (1=female, 0=male) from feature3
      - Age: numeric from feature4
      - YearsMarried: numeric from feature5
      - Religiousness: numeric from feature7
      - Education: numeric from feature8
      - Occupation: numeric from feature9
      - MaritalHappiness: numeric from feature10

    Rows with missing values in Affairs or Children or core controls are dropped.
    """
    # Work on a copy
    df = df.copy()

    # Rename and coerce columns to stable names
    # Affairs: feature2
    df['Affairs'] = pd.to_numeric(df['feature2'], errors='coerce')

    # Children: feature6 (yes/no -> 1/0). Allow various capitalizations.
    df['Children'] = df['feature6'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Gender: feature3 (female/male -> 1/0). If other encodings appear, try to map expected values.
    df['Female'] = df['feature3'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})

    # Numeric controls: coerce to numeric and keep original coding
    df['Age'] = pd.to_numeric(df['feature4'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Education'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['feature9'], errors='coerce')
    df['MaritalHappiness'] = pd.to_numeric(df['feature10'], errors='coerce')

    # Drop rows missing the dependent variable or the main independent variable
    required_cols = ['Affairs', 'Children']
    # Also drop if all controls are missing (we require at least the core controls) -- but primarily ensure core covariates are present
    required_cols += ['Female', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'MaritalHappiness']

    df = df.dropna(subset=required_cols)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Tobit (left-censored at zero) model predicting Affairs from Children and covariates.
    Also fit an OLS as a simple comparison.

    Returns a dictionary with keys:
      - 'tobit_result': fitted Tobit result (statsmodels MLE result)
      - 'ols_result': fitted OLS result (statsmodels RegressionResultsWrapper)

    Notes on the Tobit implementation: we implement a left-censoring Tobit with censor point = 0.
    The parameter vector is [beta (k), sigma]. The log-likelihood sums log(Phi) for censored obs
    and log(pdf) for uncensored obs.
    """
    import numpy as np
    from scipy import stats

    # Prepare data for modeling
    # Independent variables (controls + IV)
    exog_cols = [
        'Children',
        'Female',
        'Age',
        'YearsMarried',
        'Religiousness',
        'Education',
        'Occupation',
        'MaritalHappiness'
    ]
    X = df[exog_cols].astype(float).values
    X = sm.add_constant(X, prepend=True)  # add intercept
    y = df['Affairs'].astype(float).values

    # OLS for comparison
    ols_model = sm.OLS(y, X)
    ols_res = ols_model.fit(cov_type='HC3')

    # Tobit (left-censored at 0) implemented via GenericLikelihoodModel
    class Tobit(GenericLikelihoodModel):
        def __init__(self, endog, exog, left=0.0):
            super(Tobit, self).__init__(endog, exog)
            self.left = left

        def loglike(self, params):
            # params: [beta_0, ..., beta_k, sigma]
            params = np.asarray(params)
            k_exog = self.exog.shape[1]
            beta = params[:k_exog]
            sigma = params[k_exog]
            if sigma <= 0 or not np.isfinite(sigma):
                return -1e30
            mu = np.dot(self.exog, beta)
            y = self.endog
            cens_mask = (y <= self.left)
            uncens_mask = ~cens_mask

            ll = np.empty_like(y, dtype=float)
            # uncensored observations: log density
            if uncens_mask.any():
                z_unc = (y[uncens_mask] - mu[uncens_mask]) / sigma
                ll[uncens_mask] = -np.log(sigma) + stats.norm.logpdf(z_unc)
            # censored observations: log of CDF at (left - mu) / sigma
            if cens_mask.any():
                z_c = (self.left - mu[cens_mask]) / sigma
                # use logcdf for numerical stability
                ll[cens_mask] = stats.norm.logcdf(z_c)

            return float(ll.sum())

    # Instantiate Tobit model
    tobit_mod = Tobit(y, X, left=0.0)

    # Starting parameters: OLS betas and sigma = residual std
    start_beta = ols_res.params
    resid = ols_res.resid
    start_sigma = np.std(resid, ddof=X.shape[1]) if resid.size > 1 else 1.0
    start_params = np.r_[start_beta, start_sigma if (start_sigma > 1e-6 and np.isfinite(start_sigma)) else 1.0]

    try:
        tobit_res = tobit_mod.fit(start_params=start_params, method='bfgs', disp=False)
    except Exception:
        # Try with a different optimizer or default start params if BFGS fails
        try:
            tobit_res = tobit_mod.fit(start_params=start_params, method='Nelder-Mead', maxiter=200, disp=False)
        except Exception as e:
            # If Tobit fails to converge, return OLS only and include the exception
            return {
                'tobit_result': None,
                'ols_result': ols_res,
                'tobit_error': str(e)
            }

    return {
        'tobit_result': tobit_res,
        'ols_result': ols_res
    }