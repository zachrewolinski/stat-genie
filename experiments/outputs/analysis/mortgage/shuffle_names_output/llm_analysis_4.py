from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    - Creates a binary Approved column (1 accepted, 0 denied) using available approval/denial columns.
    - Creates a binary Female column using consumer_credit (preferred) or the female column as fallback.
    - Ensures control columns exist, coerces to numeric, imputes medians where needed, and standardizes continuous controls.
    - Drops rows missing the outcome or gender after construction.

    The returned dataframe contains the exact column names used in the model: 
      ['Approved','Female','bad_history','black','married','PI_ratio','housing_expense_ratio','loan_to_value','denied_PMI','self_employed']
    """
    df = df.copy()

    # 1) Build Approved (1 accepted, 0 denied)
    # Prefer 'mortgage_credit' if it's 1=denied,0=accepted (so Approved = 1 - mortgage_credit)
    if 'mortgage_credit' in df.columns:
        df['Approved'] = 1 - pd.to_numeric(df['mortgage_credit'], errors='coerce').astype(float)
    # Otherwise, if Unnamed: 0 is present and represents accepted=1, denied=0, use it
    elif 'Unnamed: 0' in df.columns:
        df['Approved'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce').astype(float)
    # If neither available, but there's a column 'accept' with multi-level codes, create a conservative binary:
    # treat higher values as more likely accepted (this is dataset-specific and may need review)
    elif 'accept' in df.columns:
        # convert to numeric then map to binary by thresholding at median
        tmp = pd.to_numeric(df['accept'], errors='coerce')
        thresh = tmp.median()
        df['Approved'] = (tmp >= thresh).astype(float)
    else:
        raise ValueError("No recognizable approval/denial column found. Expected 'mortgage_credit' or 'Unnamed: 0' or 'accept'.")

    # 2) Build Female indicator (1 female, 0 male)
    # Prefer 'consumer_credit' which in the provided schema is documented as female indicator
    if 'consumer_credit' in df.columns:
        df['Female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').astype(float)
    elif 'female' in df.columns:
        # 'female' in the dataset may not be perfectly binary; threshold at 0.5
        df['Female'] = (pd.to_numeric(df['female'], errors='coerce') > 0.5).astype(float)
    else:
        raise ValueError("No recognizable gender column found. Expected 'consumer_credit' or 'female'.")

    # 3) Ensure all control columns exist in the final dataframe (create with NaN if missing)
    control_cols = ['bad_history', 'black', 'married', 'PI_ratio', 'housing_expense_ratio', 'loan_to_value', 'denied_PMI', 'self_employed']
    for c in control_cols:
        if c not in df.columns:
            df[c] = np.nan

    # 4) Coerce controls to numeric and impute medians for missing values
    for c in control_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].isna().all():
            # if the column is entirely missing, fill with 0 (neutral) to allow model estimation
            df[c] = 0.0
        else:
            # impute median for remaining NaNs
            median = df[c].median()
            df[c] = df[c].fillna(median)

    # 5) Standardize continuous controls for numerical stability (in-place)
    # Choose columns expected to be continuous/ordinal
    cont_to_scale = ['PI_ratio', 'housing_expense_ratio', 'denied_PMI']
    for c in cont_to_scale:
        # if constant, leave as-is; else z-score
        if df[c].std(ddof=0) > 0:
            df[c] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
        else:
            df[c] = 0.0

    # 6) Final cleanup: drop rows with missing Approved or Female
    df = df.dropna(subset=['Approved', 'Female'])

    # Ensure binary columns are 0/1
    df['Approved'] = df['Approved'].astype(int)
    df['Female'] = df['Female'].astype(int)

    # Return only the columns needed for modeling (keeps extra columns if desired)
    final_cols = ['Approved', 'Female'] + control_cols
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (logit) to estimate the effect of applicant gender on mortgage approval,
    controlling for observed applicant and loan characteristics.

    Returns a dictionary with:
      - 'logit_results': fitted statsmodels results (with robust standard errors when possible)
      - 'marginal_effects': DataFrame of average marginal effects for predictors
      - 'predictors': list of predictors used in the model
    """
    # Select predictors (Female + all controls listed in transform)
    predictors = ['Female', 'bad_history', 'black', 'married', 'PI_ratio', 'housing_expense_ratio', 'loan_to_value', 'denied_PMI', 'self_employed']

    # Ensure predictors present in dataframe
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        raise ValueError(f"Missing predictors in dataframe: {missing}")

    X = df[predictors].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['Approved'].astype(float)

    # Fit Logit
    logit_model = sm.Logit(y, X)
    # Use try/except to catch potential convergence issues
    try:
        res = logit_model.fit(disp=False)
    except Exception:
        # retry with different method
        res = logit_model.fit(disp=False, method='bfgs', maxiter=100)

    # Try to obtain robust covariance results (HC3). Some statsmodels versions may not expose
    # a get_robustcov_results method on the results object; handle that case by computing
    # robust covariance and wrapping the original results.
    try:
        robust_res = res.get_robustcov_results(cov_type='HC3')
    except Exception:
        # compute HC3 covariance matrix and create a lightweight wrapper exposing common attributes
        try:
            from statsmodels.stats.sandwich_covariance import cov_hc3
            cov_robust = cov_hc3(res)
        except Exception:
            # as a final fallback, use the default covariance matrix
            cov_robust = res.cov_params()

        class RobustResults:
            def __init__(self, base_res, cov):
                self._res = base_res
                self.cov_robust = cov
                self.params = base_res.params
                # robust bse
                self.bse = np.sqrt(np.maximum(np.diag(cov), 0.0))
                # compute p-values based on normal approximation using robust bse
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = np.divide(self.params, self.bse, out=np.zeros_like(self.params), where=self.bse != 0)
                try:
                    pvals = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
                except Exception:
                    # fallback if scipy.stats isn't available for some reason
                    from math import erf, sqrt
                    pvals = 2 * (1 - 0.5 * (1 + np.vectorize(lambda x: erf(x / sqrt(2)))(np.abs(z))))
                self.pvalues = pvals

            def cov_params(self):
                return self.cov_robust

            def __getattr__(self, name):
                # Delegate attribute access to the original results object
                return getattr(self._res, name)

        robust_res = RobustResults(res, cov_robust)

    # Compute average marginal effects (AME)
    try:
        mfx = res.get_margeff(at='overall').summary_frame()
    except Exception:
        # if margeff fails on res, try on robust_res (which may delegate to res)
        try:
            mfx = robust_res.get_margeff(at='overall').summary_frame()
        except Exception:
            # give up and set to None if unable to compute
            mfx = None

    return {
        'logit_results': robust_res,
        'marginal_effects': mfx,
        'predictors': predictors
    }