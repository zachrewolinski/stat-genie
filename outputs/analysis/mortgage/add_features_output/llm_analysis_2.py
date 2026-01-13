from typing import Any, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hc1
from types import SimpleNamespace

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/add_features_output/mortgage.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform original dataframe for modeling gender effect on mortgage approval.

    Returns a dataframe with:
      - accept (binary outcome; kept as integer)
      - female (binary indicator; integer)
      - standardized continuous controls with *_z suffix (exact final names required)
      - binary controls coerced to int

    The function drops rows with missing values in any variable used in the model.
    """
    df = df.copy()

    # Columns required for the model (original/raw column names where appropriate)
    required_cols = [
        'accept', 'female',
        'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'married', 'self_employed', 'black', 'denied_PMI'
    ]

    # If some expected columns are not present, raise a clear error
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Subset to rows with non-missing values in the variables we will use (raw inputs)
    df = df.dropna(subset=required_cols)

    # Ensure binary/int columns are integers (0/1)
    for bcol in ['female', 'bad_history', 'married', 'self_employed', 'black', 'denied_PMI', 'accept']:
        # Some columns might be floats with 0.0/1.0; coerce to integer
        # Use astype after ensuring there are no NaNs
        df[bcol] = df[bcol].astype(int)

    # Continuous controls mapping: raw column -> required final z column name
    cont_map: Dict[str, str] = {
        'mortgage_credit': 'mortgage_credit_z',
        'consumer_credit': 'consumer_credit_z',
        'PI_ratio': 'PI_ratio_z',
        'loan_to_value': 'loan_to_value_z',
        'housing_expense_ratio': 'housing_exp_ratio_z'  # final required name differs from raw
    }

    # Clip unreasonable loan_to_value outliers (keeps values within a reasonable bound, e.g., 0 to 2)
    df['loan_to_value'] = pd.to_numeric(df['loan_to_value'], errors='coerce')
    df = df.dropna(subset=['loan_to_value'])
    df['loan_to_value'] = df['loan_to_value'].clip(lower=0.0, upper=2.0)

    # Standardize continuous variables and add *_z columns used in the model
    for raw_col, z_col in cont_map.items():
        # ensure numeric
        df[raw_col] = pd.to_numeric(df[raw_col], errors='coerce')
        # If any remaining NaNs appear after coercion, drop them
        df = df.dropna(subset=[raw_col])
        mean = df[raw_col].mean()
        std = df[raw_col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # if no variation, create 0 column
            df[z_col] = 0.0
        else:
            df[z_col] = (df[raw_col] - mean) / std

    # Final safety drop in case any NA remained in the required final columns
    final_cols = ['accept', 'female', 'mortgage_credit_z', 'consumer_credit_z', 'bad_history',
                  'PI_ratio_z', 'loan_to_value_z', 'housing_exp_ratio_z',
                  'married', 'self_employed', 'black', 'denied_PMI']
    df = df.dropna(subset=final_cols)

    # Return dataframe that contains the columns the model expects (keep original plus z cols)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression of mortgage acceptance on female with controls.

    Model specification (logit):
      accept ~ female + mortgage_credit_z + consumer_credit_z + bad_history
               + PI_ratio_z + loan_to_value_z + housing_exp_ratio_z
               + married + self_employed + black + denied_PMI

    Returns a dictionary containing:
      - 'results_robust': the statsmodels fitted results object with robust (HC1) covariance
      - 'female_margeff': a one-row pandas Series with the average marginal effect for the female indicator (if available)

    Notes:
      - Robust (HC1) standard errors are used for inference.
      - The function expects the dataframe produced by the transform() function above.
    """
    # columns used in the model (must match transform output)
    X_cols: List[str] = [
        'female',
        'mortgage_credit_z', 'consumer_credit_z', 'bad_history',
        'PI_ratio_z', 'loan_to_value_z', 'housing_exp_ratio_z',
        'married', 'self_employed', 'black', 'denied_PMI'
    ]

    # Safety check
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing columns needed for modeling: {missing}")

    X = df[X_cols].copy()
    X = sm.add_constant(X)
    y = df['accept']

    # Fit logistic regression
    logit_res = sm.Logit(y, X).fit(disp=False)

    # Obtain robust covariance (HC1). Some versions of statsmodels may not provide
    # get_robustcov_results on the results object; compute HC1 sandwich covariance directly.
    try:
        cov = cov_hc1(logit_res)
        cov_df = pd.DataFrame(cov, index=logit_res.params.index, columns=logit_res.params.index)
        bse_robust = pd.Series(np.sqrt(np.diag(cov_df)), index=logit_res.params.index)

        # Create a lightweight results-like object that exposes params, bse, and cov_params()
        class RobustResults(SimpleNamespace):
            def __init__(self, params, bse, cov_df, orig_res):
                super().__init__()
                self.params = params
                self.bse = bse
                self._cov = cov_df
                self.orig_res = orig_res  # store original for any further use

            def cov_params(self):
                return self._cov

            # provide summary-like minimal info if needed
            @property
            def df_model(self):
                return getattr(self.orig_res, 'df_model', None)

            @property
            def df_resid(self):
                return getattr(self.orig_res, 'df_resid', None)

        results_robust = RobustResults(logit_res.params, bse_robust, cov_df, logit_res)
    except Exception:
        # Fallback: if HC1 computation fails for some reason, return the original results object
        results_robust = logit_res

    # Compute average marginal effects (if possible) and extract the female effect
    female_margeff = None
    try:
        margeff = logit_res.get_margeff(at='overall')
        mf_frame = margeff.summary_frame()
        if 'female' in mf_frame.index:
            female_margeff = mf_frame.loc['female']
        else:
            female_margeff = None
    except Exception:
        female_margeff = None

    return {
        'results_robust': results_robust,
        'female_margeff': female_margeff
    }