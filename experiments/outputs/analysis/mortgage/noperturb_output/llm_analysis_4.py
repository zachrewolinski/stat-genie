from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/noperturb_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for modeling:
    - Select relevant columns
    - Drop rows with missing values in predictors/outcome
    - Ensure binary flags are integer type
    - Create standardized (z-scored) versions of continuous/ordinal controls to aid model stability and interpretation
    - Return dataframe containing the final columns used in the model
    Final columns returned (at minimum):
      ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
       'z_PI_ratio', 'z_loan_to_value', 'z_housing_expense_ratio', 'z_mortgage_credit', 'z_consumer_credit']
    """

    # Work on a copy
    df = df.copy()

    # Columns of interest (based on dataset schema)
    cols = [
        'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]

    # Keep only these columns (if any are missing this will raise KeyError so better to intersect)
    cols_present = [c for c in cols if c in df.columns]
    df = df[cols_present]

    # Drop rows with missing values in any of the vars we need
    df = df.dropna(subset=cols_present)

    # Ensure binary flags are integers (0/1)
    binary_cols = [c for c in ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI'] if c in df.columns]
    for c in binary_cols:
        # cast to int when possible (if floats like 0.0/1.0)
        # If values are boolean, .astype(int) will work as well.
        try:
            df[c] = df[c].astype(int)
        except Exception:
            # If casting to int fails (e.g., strings), try to map common representations
            mapping = {'Y': 1, 'y': 1, 'yes': 1, 'Yes': 1, 'TRUE': 1, 'True': 1, 'true': 1,
                       'N': 0, 'n': 0, 'no': 0, 'No': 0, 'FALSE': 0, 'False': 0, 'false': 0}
            df[c] = df[c].map(mapping).fillna(0).astype(int)

    # Standardize continuous and ordinal predictors to mean 0, sd 1 (population sd ddof=0)
    to_z = [c for c in ['PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'mortgage_credit', 'consumer_credit'] if c in df.columns]
    for c in to_z:
        zname = 'z_' + c if not c.startswith('z_') else c
        # avoid division by zero
        std = df[c].std(ddof=0)
        mean = df[c].mean()
        if std == 0 or np.isnan(std):
            df[zname] = 0.0
        else:
            df[zname] = (df[c] - mean) / std

    # Keep only the final set of columns used in the model (original binaries + z- columns)
    final_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'z_PI_ratio', 'z_loan_to_value', 'z_housing_expense_ratio', 'z_mortgage_credit', 'z_consumer_credit'
    ]
    # Some z_ columns might not have been created if original columns were missing; filter
    final_cols = [c for c in final_cols if c in df.columns]

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting acceptance (accept = 1) from female (primary IV)
    and controls. Returns the fitted model with robust standard errors, estimated odds ratios
    and confidence intervals, and a textual summary.

    The function expects that `df` is the output of transform() and contains the columns
    described in the conceptual variables.
    """

    # Ensure required libs are available
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hc3
    from scipy.stats import norm

    # Define predictor list (use standardized versions where created)
    predictors = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'z_PI_ratio',
        'z_loan_to_value',
        'z_housing_expense_ratio',
        'z_mortgage_credit',
        'z_consumer_credit'
    ]
    # Keep only predictors present in df (in case some controls were missing from dataset)
    predictors = [p for p in predictors if p in df.columns]

    if 'accept' not in df.columns:
        raise ValueError("Dependent variable 'accept' not found in dataframe")

    X = df[predictors].astype(float) if predictors else pd.DataFrame(index=df.index)
    X = sm.add_constant(X, has_constant='add')
    y = df['accept'].astype(float)

    # Fit logistic regression
    model_sm = sm.Logit(y, X)
    try:
        fit = model_sm.fit(disp=False, method='lbfgs', maxiter=200)
    except Exception:
        # Fallback to default fit (may warn) if lbfgs fails
        fit = model_sm.fit(disp=False)

    # Compute robust (heteroskedasticity-consistent) covariance (HC3)
    try:
        cov_robust = cov_hc3(fit)
    except Exception:
        # If cov_hc3 fails for some reason, fall back to the model's covariance
        cov_robust = fit.cov_params()

    # Robust standard errors
    robust_bse = np.sqrt(np.diag(cov_robust))

    # Parameters
    params = fit.params
    # 95% z critical value
    z = norm.ppf(0.975)

    # Compute confidence intervals using robust standard errors
    ci_lower = params - z * robust_bse
    ci_upper = params + z * robust_bse

    # Compute odds ratios and CIs
    or_vals = np.exp(params)
    conf_or_lower = np.exp(ci_lower)
    conf_or_upper = np.exp(ci_upper)

    odds_df = pd.DataFrame({
        'OR': or_vals,
        'CI_lower': conf_or_lower,
        'CI_upper': conf_or_upper
    }, index=params.index)

    # Attach robust covariance and robust bse to the fit object for downstream inspection
    try:
        setattr(fit, 'cov_robust', cov_robust)
        setattr(fit, 'bse_robust', pd.Series(robust_bse, index=params.index))
    except Exception:
        # If setting attributes fails, continue without attaching
        pass

    # Prepare textual summary: include the original summary and append robust SEs/ORs table
    try:
        base_summary = fit.summary().as_text()
    except Exception:
        base_summary = str(fit)

    summary_text = base_summary + '\n\nRobust (HC3) standard errors and Odds Ratios (95% CI):\n' + odds_df.to_string(float_format=lambda x: f"{x:.4f}")

    results = {
        'fit_robust': fit,
        'odds_ratios': odds_df,
        'summary_text': summary_text
    }

    return results