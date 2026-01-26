from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (Psychology Today) dataset into a clean dataframe for modeling.

    Produces the following columns (used in the model):
      - AffairFreq: numeric - derived from original 'education' column (which, per schema, encodes extramarital affair frequency)
      - AffairBinary: 0/1 indicator (1 if AffairFreq > 0)
      - Children: 0/1 (1 = children present in marriage, from original 'age' column which encodes children yes/no)
      - Gender_Male: 0/1 (1 = male, from original 'children' column which actually contains gender)
      - Age: numeric (from original 'rating' column)
      - YearsMarried: numeric (from original 'gender' column according to schema description)
      - EducationLevel: numeric (from original 'affairs' column per schema description)
      - Religiousness: numeric (original column 'religiousness')
      - Occupation: numeric (original column 'occupation')
      - MarriageHappiness: numeric (from original 'rownames')

    The function handles basic type conversions and drops rows missing the primary variables.
    """
    # Work on a copy
    df = df.copy()

    # Map the variable names according to schema-annotations (several column descriptions appear swapped in the provided schema)
    # Primary DV: frequency of extramarital affairs is stored in 'education' (per schema description)
    df['AffairFreq'] = pd.to_numeric(df.get('education'), errors='coerce')
    # Create binary indicator for any affair
    df['AffairBinary'] = (df['AffairFreq'] > 0).astype(int)

    # IV: presence of children. Per schema, column 'age' encodes presence of children (factor yes/no)
    # Normalize and map strings to 1/0; handle a few variants.
    def map_children_val(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ['yes', 'y', '1', 'true', 't']:
            return 1
        if s in ['no', 'n', '0', 'false', 'f']:
            return 0
        # if it looks like boolean stored as numeric
        try:
            xi = float(s)
            if xi == 1:
                return 1
            if xi == 0:
                return 0
        except Exception:
            pass
        return np.nan

    df['Children'] = df.get('age').map(map_children_val)

    # Gender: per schema the column named 'children' actually contains gender labels
    # Map male->1 female->0; if other coding, try to respond reasonably
    def map_gender_to_male(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ['male', 'm', 'man']:
            return 1
        if s in ['female', 'f', 'woman']:
            return 0
        # try numeric
        try:
            xi = float(s)
            # if it's coded numerically, we cannot be sure; keep as nan
            return np.nan
        except Exception:
            return np.nan

    df['Gender_Male'] = df.get('children').map(map_gender_to_male)

    # Age (numeric) is stored in 'rating' per schema
    df['Age'] = pd.to_numeric(df.get('rating'), errors='coerce')

    # Years married - per schema the column 'gender' contains years married numeric codes
    df['YearsMarried'] = pd.to_numeric(df.get('gender'), errors='coerce')

    # EducationLevel is stored in column named 'affairs' per schema mismatch
    df['EducationLevel'] = pd.to_numeric(df.get('affairs'), errors='coerce')

    # Religiousness appears to be correctly named
    df['Religiousness'] = pd.to_numeric(df.get('religiousness'), errors='coerce')

    # Occupation code
    df['Occupation'] = pd.to_numeric(df.get('occupation'), errors='coerce')

    # Marriage happiness: original column 'rownames' encodes self-rating of marriage (1..5)
    df['MarriageHappiness'] = pd.to_numeric(df.get('rownames'), errors='coerce')

    # Keep only the columns we will use in modeling
    keep_cols = ['AffairFreq', 'AffairBinary', 'Children', 'Gender_Male', 'Age', 'YearsMarried',
                 'EducationLevel', 'Religiousness', 'Occupation', 'MarriageHappiness']
    df = df[keep_cols]

    # Drop observations missing the primary dependent variable or the primary independent variable
    df = df.dropna(subset=['AffairFreq', 'Children'])

    # Optionally: fill some control missings with medians (we keep them as NaN for modeling which will drop rows as necessary)
    # But to keep sample larger we impute simple median for numerical controls (safe default). This is optional and noted here.
    numeric_controls = ['Gender_Male', 'Age', 'YearsMarried', 'EducationLevel', 'Religiousness', 'Occupation', 'MarriageHappiness']
    for col in numeric_controls:
        if col in df.columns:
            # if the column is very sparse we keep NaNs; otherwise fill median
            if df[col].notna().sum() >= (0.5 * len(df)):
                df[col] = df[col].fillna(df[col].median())

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two models to estimate the effect of having children on engagement in extramarital affairs:
      1) Tobit (censored) model with left-censoring at 0 on AffairFreq (primary analysis),
      2) Logistic regression on AffairBinary (robustness: any affair vs none).

    Returns a dictionary with the fitted parameters and summaries.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import optimize, stats

    # Select variables and drop rows with missing values in model matrix
    model_vars = ['AffairFreq', 'AffairBinary', 'Children', 'Gender_Male', 'Age', 'EducationLevel',
                  'Religiousness', 'YearsMarried', 'Occupation', 'MarriageHappiness']
    data = df[model_vars].dropna()

    # Design matrix for controls + IV
    X_cols = ['Children', 'Gender_Male', 'Age', 'EducationLevel', 'Religiousness', 'YearsMarried',
              'Occupation', 'MarriageHappiness']
    X = data[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')  # add intercept
    y = data['AffairFreq'].astype(float)

    # ------------------
    # Tobit (censored at 0) via MLE
    # ------------------
    # Tobit log-likelihood: for y_i == 0 -> log Phi(-(Xb)/sigma), for y_i > 0 -> log (1/sigma * phi((y - Xb)/sigma))
    def tobit_negloglik(params, y, X):
        # params: [beta_0 ... beta_k, log_sigma]
        k = X.shape[1]
        betas = params[:k]
        log_sigma = params[k]
        sigma = np.exp(log_sigma)
        XB = X.values.dot(betas)
        # Observed positives
        mask_pos = (y > 0).values
        y_pos = y.values[mask_pos]
        XB_pos = XB[mask_pos]
        # log pdf for positives
        ll_pos = stats.norm.logpdf((y_pos - XB_pos) / sigma) - log_sigma
        # zeros
        mask_zero = ~mask_pos
        XB_zero = XB[mask_zero]
        # log cdf for zeros: log Phi( - XB / sigma )
        ll_zero = stats.norm.logcdf(- XB_zero / sigma)
        # sum negative
        total_ll = ll_pos.sum() + ll_zero.sum()
        return -total_ll

    # Starting values: OLS on positive observations
    pos_mask = y > 0
    if pos_mask.sum() >= X.shape[1]:
        ols_start = sm.OLS(y[pos_mask], X.loc[pos_mask]).fit()
        start_betas = ols_start.params.values
        start_sigma = np.log(np.std(ols_start.resid.dropna()))
    else:
        # fallback small init
        start_betas = np.zeros(X.shape[1])
        start_sigma = np.log(max(1.0, y.std()))

    start_params = np.concatenate([start_betas, [start_sigma]])

    # Optimize negative log-likelihood
    opt_res = optimize.minimize(
        fun=tobit_negloglik,
        x0=start_params,
        args=(y, X),
        method='BFGS',
        options={'disp': False, 'maxiter': 1000}
    )

    # Collect results
    if not opt_res.success:
        # Try a more robust method with bounds on sigma
        def wrapper(params):
            return tobit_negloglik(params, y, X)
        bounds = [(None, None)] * (X.shape[1]) + [(None, None)]
        opt_res = optimize.minimize(wrapper, start_params, method='L-BFGS-B', bounds=bounds)

    params_hat = opt_res.x
    k = X.shape[1]
    beta_hat = params_hat[:k]
    sigma_hat = float(np.exp(params_hat[k]))

    # Approximate covariance using inverse Hessian if available
    try:
        hess_inv = opt_res.hess_inv
        # For BFGS, hess_inv is an ndarray
        if hasattr(hess_inv, 'todense'):
            cov = np.array(hess_inv.todense())
        else:
            cov = np.array(hess_inv)
        se_params = np.sqrt(np.diag(cov))
    except Exception:
        cov = None
        se_params = np.full_like(params_hat, np.nan)

    # Prepare a readable output for Tobit
    tobit_result = {
        'params': dict(zip(['const'] + X_cols + ['log_sigma'], params_hat.tolist())),
        'sigma': sigma_hat,
        'se': dict(zip(['const'] + X_cols + ['log_sigma'], se_params.tolist())),
        'neg_loglik': float(opt_res.fun),
        'converged': bool(opt_res.success),
        'nobs': int(len(y))
    }

    # Compute AIC/BIC
    p = len(params_hat)
    llf = -opt_res.fun
    tobit_result['AIC'] = 2 * p - 2 * llf
    tobit_result['BIC'] = np.log(len(y)) * p - 2 * llf

    # ------------------
    # Logistic regression (robustness): Any affair vs none
    # ------------------
    try:
        logit_X = X.copy()
        logit_y = data['AffairBinary'].astype(int)
        logit_model = sm.Logit(logit_y, logit_X)
        logit_res = logit_model.fit(disp=False)
        logit_summary = logit_res.summary().as_text()
    except Exception as e:
        logit_res = None
        logit_summary = f'Logit failed: {e}'

    results = {
        'tobit': tobit_result,
        'logit_summary_text': logit_summary,
        'logit_result_obj': logit_res,
        'design_matrix_columns': ['const'] + X_cols,
        'nobs': int(len(data))
    }

    return results


