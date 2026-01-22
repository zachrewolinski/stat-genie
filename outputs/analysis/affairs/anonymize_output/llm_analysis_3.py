from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/anonymize_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Psychology Today / Fair (1978) dataset into a dataframe ready for modeling.

    Produces the following columns used by the modeling function:
      - AffairFreq: numeric reported frequency (left-censored at 0)
      - HasChildren: binary (1=yes, 0=no)
      - Female: binary (1=female, 0=male)
      - Age, YearsMarried, Religiosity, Education, Occupation, MaritalHappiness: numeric controls

    Drops observations with missing values in any of the required columns.
    """
    df = df.copy()

    # Rename raw columns to meaningful names (we keep original columns untouched by renaming into new ones)
    # Original schema: feature2 = affair frequency, feature3 = gender, feature4 = age, feature5 = years married,
    # feature6 = children (yes/no), feature7 = religiosity, feature8 = education, feature9 = occupation, feature10 = marriage rating
    df = df.rename(columns={
        'feature2': 'AffairFreq_raw',
        'feature3': 'Gender',
        'feature4': 'Age',
        'feature5': 'YearsMarried',
        'feature6': 'Children',
        'feature7': 'Religiosity',
        'feature8': 'Education',
        'feature9': 'Occupation',
        'feature10': 'MaritalHappiness'
    })

    # Coerce affair frequency to numeric. The dataset uses special codes (0,1,2,3,7,12); keep values as provided.
    df['AffairFreq'] = pd.to_numeric(df.get('AffairFreq_raw'), errors='coerce')

    # Map Children -> HasChildren (1 = 'yes', 0 = otherwise)
    df['HasChildren'] = df['Children'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # Map Gender -> Female (1 = 'female', 0 = otherwise)
    df['Female'] = df['Gender'].apply(lambda x: 1 if str(x).strip().lower() == 'female' else 0)

    # Coerce control variables to numeric (some are already numeric, but enforce dtype and coerce errors to NaN)
    for col in ['Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']:
        df[col] = pd.to_numeric(df.get(col), errors='coerce')

    # Drop rows with missing values in any variable needed for the model
    required = ['AffairFreq', 'HasChildren', 'Female', 'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']
    df = df.dropna(subset=required)

    # Ensure AffairFreq is non-negative; set any negative values to zero (defensive)
    df.loc[df['AffairFreq'] < 0, 'AffairFreq'] = 0

    # Keep only columns needed for modeling plus originals for traceability
    keep_cols = required + ['AffairFreq_raw', 'Children', 'Gender']
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a Tobit (left-censored at 0) model for AffairFreq on HasChildren and controls.

    Implementation notes:
    - We maximize the Tobit log-likelihood via scipy.optimize.minimize (BFGS).
    - Parameters are [beta (k), log_sigma]. We exponentiate log_sigma to ensure sigma>0.
    - Returns a dictionary with parameter table, sigma, log-likelihood, and optimizer info.
    """
    import numpy as np
    from scipy import stats, optimize
    import pandas as pd

    # Prepare X and y
    exog_cols = ['HasChildren', 'Female', 'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']
    X = df[exog_cols].astype(float)
    X = sm.add_constant(X)  # adds 'const' column
    y = df['AffairFreq'].astype(float)

    # Censoring point (left-censor at 0)
    censor = 0.0

    # Initial values: OLS on uncensored observations
    uncensored_mask = (y > censor)
    if uncensored_mask.sum() < X.shape[1]:
        # Not enough uncensored observations to estimate OLS; use simple inits
        init_beta = np.zeros(X.shape[1])
        init_log_sigma = np.log(max(y.std(), 1.0))
    else:
        try:
            ols_coef, _, _, _ = np.linalg.lstsq(X.loc[uncensored_mask].values, y.loc[uncensored_mask].values, rcond=None)
            init_beta = ols_coef
            resid = y.loc[uncensored_mask].values - X.loc[uncensored_mask].values.dot(init_beta)
            init_log_sigma = np.log(resid.std() if resid.size > 1 else max(1.0, y.std()))
        except Exception:
            init_beta = np.zeros(X.shape[1])
            init_log_sigma = np.log(max(y.std(), 1.0))

    init_params = np.concatenate([init_beta, [init_log_sigma]])

    # Negative log-likelihood for Tobit (left-censor at 'censor')
    def neg_loglik(params):
        beta = params[:-1]
        sigma = np.exp(params[-1])
        xb = X.values.dot(beta)

        # For uncensored observations (y > censor): use normal density
        z_unc = (y.values - xb) / sigma
        ll_unc = -np.log(sigma) + stats.norm.logpdf(z_unc)

        # For censored observations (y == censor): probability mass is Phi((censor - xb)/sigma)
        z_c = (censor - xb) / sigma
        # Use logcdf for numerical stability
        ll_c = stats.norm.logcdf(z_c)

        ll = np.where(y.values > censor, ll_unc, ll_c)
        # Return negative sum log-likelihood
        return -np.sum(ll)

    # Minimize negative log-likelihood
    opt_res = optimize.minimize(neg_loglik, init_params, method='BFGS')

    # Collect estimates
    est = opt_res.x
    beta_est = est[:-1]
    log_sigma_est = est[-1]
    sigma_est = float(np.exp(log_sigma_est))

    # Compute standard errors using inverse Hessian approximation returned by BFGS
    if hasattr(opt_res, 'hess_inv') and opt_res.hess_inv is not None:
        hess_inv = opt_res.hess_inv
        try:
            # hess_inv may be a numpy array or an object with todense()
            cov = hess_inv if isinstance(hess_inv, np.ndarray) else np.array(hess_inv.todense())
            se = np.sqrt(np.abs(np.diag(cov)))
        except Exception:
            cov = None
            se = np.full_like(est, np.nan, dtype=float)
    else:
        cov = None
        se = np.full_like(est, np.nan, dtype=float)

    # Prepare parameter table
    param_names = list(X.columns) + ['log_sigma']
    results_table = pd.DataFrame({
        'param': est,
        'se': se
    }, index=param_names)

    # z-statistics and p-values (normal approx)
    results_table['z'] = results_table['param'] / results_table['se']
    results_table['pval'] = 2 * stats.norm.sf(np.abs(results_table['z']))

    # Adjust sigma se via delta method if log_sigma se available
    if not np.isnan(se[-1]):
        se_sigma = se[-1] * sigma_est
    else:
        se_sigma = np.nan

    # Add sigma (on original scale) to output
    results_table.loc['sigma', :] = [sigma_est, se_sigma, (sigma_est / se_sigma) if se_sigma and not np.isnan(se_sigma) else np.nan, np.nan]

    output = {
        'params_table': results_table,
        'cov_matrix_approx': cov,
        'log_likelihood': -opt_res.fun,
        'converged': bool(opt_res.success),
        'message': opt_res.message,
        'nobs': int(df.shape[0])
    }

    return output


