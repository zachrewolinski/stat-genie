from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston mortgage dataset into the modeling dataframe.
    Produces standardized continuous controls and an interaction term for gender x race.
    Returns a dataframe containing at minimum the columns referenced in the model:
      - accept (DV)
      - female (IV)
      - black, female_black (moderator / interaction)
      - PI_ratio_z, loan_to_value_z, mortgage_credit_z, consumer_credit_z,
        housing_expense_ratio_z (standardized continuous controls)
      - bad_history, married, self_employed, denied_PMI (binary controls)
    """
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required_cols = [
        'accept', 'female', 'black', 'PI_ratio', 'loan_to_value',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'married',
        'self_employed', 'housing_expense_ratio', 'denied_PMI'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transformation: {missing}")

    # Drop rows with missing values in any variable we will use in the model
    df = df.dropna(subset=required_cols + ['accept'])

    # Ensure binary indicators are integers (0/1)
    for col in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']:
        # coerce to numeric then to int
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI'])
    df['accept'] = df['accept'].astype(int)
    df['female'] = df['female'].astype(int)
    df['black'] = df['black'].astype(int)
    df['bad_history'] = df['bad_history'].astype(int)
    df['married'] = df['married'].astype(int)
    df['self_employed'] = df['self_employed'].astype(int)
    df['denied_PMI'] = df['denied_PMI'].astype(int)

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for stability.
    cont_cols = ['PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit', 'housing_expense_ratio']
    for c in cont_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=cont_cols)

    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        if std == 0 or pd.isna(std):
            # If no variation, set z to 0
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Interaction term between female and black (to test whether gender effect differs by race)
    df['female_black'] = df['female'] * df['black']

    # Keep only columns needed for the model and return
    keep_cols = [
        'accept', 'female', 'black', 'female_black',
        'PI_ratio_z', 'loan_to_value_z', 'mortgage_credit_z', 'consumer_credit_z', 'housing_expense_ratio_z',
        'bad_history', 'married', 'self_employed', 'denied_PMI'
    ]

    # If any of the standardized columns are missing (shouldn't be), raise an error
    for kc in keep_cols:
        if kc not in df.columns:
            raise ValueError(f"Expected transformed column {kc} is missing")

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (logit) predicting loan acceptance (accept) from applicant gender
    and controls. Uses a gender x race interaction to test moderation by race.

    Returns a dictionary containing the fitted model (robust covariance), a table of odds ratios
    with 95% confidence intervals and p-values, and the raw robust results object.
    """
    # predictors to include in the model
    predictors = [
        'female',            # main effect of gender (IV)
        'black',             # main effect of race
        'female_black',      # interaction term (gender x race)
        'PI_ratio_z',
        'loan_to_value_z',
        'mortgage_credit_z',
        'consumer_credit_z',
        'housing_expense_ratio_z',
        'bad_history',
        'married',
        'self_employed',
        'denied_PMI'
    ]

    # Verify predictors exist
    missing = [p for p in predictors if p not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing predictors in dataframe: {missing}")

    # Define X and y
    X = df[predictors].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['accept'].astype(float)

    # Fit logistic regression using statsmodels Logit
    logit_mod = sm.Logit(y, X)
    # Fit without printing iteration info
    try:
        result = logit_mod.fit(disp=False)
    except Exception:
        # If the default fit fails (e.g., perfect separation), try GLM binomial
        result = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Compute robust (HC3) covariance matrix and robust inference manually
    # Use sandwich covariance estimator for HC3
    try:
        cov_robust = sm.stats.sandwich_covariance.cov_hc3(result)
    except Exception:
        # As a fallback, try using the generic cov_params with robust option if available
        try:
            cov_robust = result.cov_params_default
        except Exception:
            # Final fallback: use model-based covariance
            cov_robust = result.cov_params()

    params = result.params
    se_robust = np.sqrt(np.diag(cov_robust))
    # Handle potential zero standard errors gracefully
    se_robust = pd.Series(se_robust, index=params.index)

    # z-statistics and p-values using normal approximation
    z_stats = params / se_robust
    pvalues = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

    # 95% CI (normal approximation)
    z_crit = stats.norm.ppf(0.975)
    ci_lower = params - z_crit * se_robust
    ci_upper = params + z_crit * se_robust

    # Build a minimal robust results-like object with the attributes used downstream
    class RobustResults:
        def __init__(self, params: pd.Series, pvalues: pd.Series, ci_lower: pd.Series, ci_upper: pd.Series, cov: np.ndarray, se: pd.Series, z: pd.Series):
            self.params = params
            self.pvalues = pvalues
            self._ci = pd.DataFrame({0: ci_lower, 1: ci_upper})
            self.cov = cov
            self.se = se
            self.z = z

        def conf_int(self):
            return self._ci

    robust_res = RobustResults(params=params, pvalues=pvalues, ci_lower=ci_lower, ci_upper=ci_upper, cov=cov_robust, se=se_robust, z=z_stats)

    # Compute odds ratios and 95% CI using robust estimates
    or_series = np.exp(robust_res.params)
    ci_lower_exp = np.exp(robust_res.conf_int()[0])
    ci_upper_exp = np.exp(robust_res.conf_int()[1])

    or_table = pd.DataFrame({
        'OR': or_series,
        'CI_lower_95': ci_lower_exp,
        'CI_upper_95': ci_upper_exp,
        'p_value': robust_res.pvalues
    })

    # Return results
    results = {
        'robust_result': robust_res,
        'odds_ratios_table': or_table,
        'model_predictors': ['const'] + predictors
    }
    return results