from typing import Any
import numpy as np
import pandas as pd
import sklearn  # kept for compatibility if downstream code expects it
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
from statsmodels.stats.sandwich_covariance import cov_hc3
from scipy.stats import norm

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston mortgage dataset into an analysis-ready dataframe.

    Steps:
    - Require that all raw columns needed for transformation are present.
    - Drop rows with missing values on those columns (complete-case analysis).
    - Ensure binary indicators are integer-typed (0/1).
    - Standardize numeric continuous controls (z-score) and create new columns with suffix '_std'.

    Returns the transformed dataframe containing all columns listed in the conceptual variables.
    """
    df = df.copy()

    # Raw columns required to produce the final analysis variables
    required_raw_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio'
    ]

    # Ensure all required raw columns are present
    missing_raw = [c for c in required_raw_cols if c not in df.columns]
    if missing_raw:
        raise KeyError(f"Missing required raw columns for transform: {missing_raw}")

    # Keep only required raw columns
    df = df.loc[:, required_raw_cols].copy()

    # Drop rows missing any required raw column values (complete-case)
    df = df.dropna(subset=required_raw_cols)

    # Ensure binary columns are integer-coded 0/1
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history']
    for col in binary_cols:
        # Safely convert booleans or numeric-like to integers; if values are non-numeric strings, this will raise
        if pd.api.types.is_bool_dtype(df[col].dtype):
            df[col] = df[col].astype(int)
        else:
            df[col] = pd.to_numeric(df[col], errors='raise').astype(int)

    # Standardize continuous control variables and store as new columns with suffix '_std'
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in cont_cols:
        mu = df[col].mean()
        sigma = df[col].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            df[col + '_std'] = 0.0
        else:
            df[col + '_std'] = (df[col] - mu) / sigma

    # Final columns required by the conceptual variables (must exist in final dataframe)
    final_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std', 'loan_to_value_std', 'housing_expense_ratio_std'
    ]

    # Verify all final cols were created
    missing_final = [c for c in final_cols if c not in df.columns]
    if missing_final:
        raise KeyError(f"Failed to produce required final columns: {missing_final}")

    # Return only the final columns in the specified order
    return df.loc[:, final_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability of mortgage acceptance (accept=1) from applicant gender (female)
    while adjusting for a set of controls capturing creditworthiness and demographics.

    Returns a dictionary with the fitted model, a robust results wrapper (via robust covariance), an estimated average marginal effect for being female,
    and a small summary with the female coefficient odds ratio and robust 95% CI.
    """
    df = df.copy()

    # Define model covariates (must match columns created in transform)
    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std', 'loan_to_value_std', 'housing_expense_ratio_std'
    ]

    # Ensure all required covariates and outcome exist in df
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns required for modeling: {missing}")

    # Prepare design matrices
    X = sm.add_constant(df[X_cols], has_constant='add')
    y = df['accept'].astype(float)

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False, maxiter=200)

    # Compute robust covariance matrix. Prefer get_robustcov_results when available,
    # otherwise compute sandwich robust covariance manually using the score-based formulation.
    try:
        # Try to use get_robustcov_results if available (newer versions)
        robust_res = res.get_robustcov_results(cov_type='HC3')
        cov_robust = robust_res.cov_params()
    except Exception:
        # Fallback: compute sandwich (robust) covariance manually.
        # For Logit, score for observation i is x_i * (y_i - p_i).
        # Sandwich covariance = inv(X' W X) @ (X' diag((y-p)^2) X) @ inv(X' W X)
        p = res.predict(X)
        resid = (y - p).to_numpy().reshape(-1)
        X_mat = np.asarray(X)
        # meat = X' diag(resid^2) X
        meat = X_mat.T.dot(np.diag(resid ** 2)).dot(X_mat)
        # bread_inv: model-based covariance (inverse Fisher information)
        bread_inv = np.asarray(res.cov_params())
        cov_mat = bread_inv.dot(meat).dot(bread_inv)
        cov_robust = pd.DataFrame(cov_mat, index=res.params.index, columns=res.params.index)

    # robust standard errors as a pandas Series aligned with parameter names
    robust_se = pd.Series(np.sqrt(np.diag(cov_robust)), index=res.params.index)

    # Compute Average Marginal Effect (AME) of female by computing predicted probabilities when female=1 vs female=0
    X_f1 = X.copy()
    X_f0 = X.copy()
    X_f1['female'] = 1
    X_f0['female'] = 0
    pred_f1 = res.predict(X_f1)
    pred_f0 = res.predict(X_f0)
    avg_marginal_effect_female = (pred_f1 - pred_f0).mean()

    # Odds ratio and robust 95% CI for the female coefficient (use normal approximation with robust SE)
    coef = res.params['female']
    se = float(robust_se['female'])
    z = norm.ppf(0.975)
    ci_lower = coef - z * se
    ci_upper = coef + z * se
    or_point = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    summary = {
        'n_obs': int(df.shape[0]),
        'female_coef_mle': float(coef),
        'female_se_robust': se,
        'female_avg_marginal_effect': float(avg_marginal_effect_female),
        'female_odds_ratio': or_point,
        'female_odds_ratio_robust_CI_95': [or_ci_lower, or_ci_upper]
    }

    results = {
        'mle_result': res,
        'robust_cov': cov_robust,
        'robust_se': robust_se,
        'avg_marginal_effect_female': avg_marginal_effect_female,
        'female_odds_ratio': or_point,
        'female_odds_ratio_robust_CI_95': [or_ci_lower, or_ci_upper],
        'summary_dict': summary
    }

    return results