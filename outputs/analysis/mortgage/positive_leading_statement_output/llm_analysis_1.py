from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from types import SimpleNamespace
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_hc3  # kept for compatibility but may not be used


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the Boston Fed mortgage dataset for logistic regression.
    Produces standardized continuous controls and ensures binary variables are integer typed.

    Output columns used in modeling:
      - female, accept, black, bad_history, married, self_employed
      - mortgage_credit_std, consumer_credit_std, PI_ratio_std, loan_to_value_std, housing_expense_ratio_std
    """
    df = df.copy()

    # Columns required for analysis (raw names expected in input)
    required_cols = [
        'female', 'accept', 'black', 'housing_expense_ratio', 'self_employed',
        'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value'
    ]

    # Coerce numeric where appropriate
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Ensure binary variables are integer 0/1
    binary_cols = ['female', 'accept', 'black', 'bad_history', 'married', 'self_employed']
    for b in binary_cols:
        # Round then convert to int to guard against floats like 0.0/1.0
        df[b] = df[b].round(0).astype(int)

    # Continuous controls to standardize (mean 0, std 1). Use population std (ddof=0) for stability.
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # If std is zero (unlikely), avoid division by zero
        if std == 0 or np.isnan(std):
            df[c + '_std'] = 0.0
        else:
            df[c + '_std'] = (df[c] - mean) / std

    # Keep only columns necessary for modelling + dependent variable
    cols_to_keep = [
        'female', 'accept', 'black', 'bad_history', 'married', 'self_employed',
        'mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std', 'loan_to_value_std', 'housing_expense_ratio_std'
    ]

    df = df[cols_to_keep].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression of mortgage acceptance on gender (female) controlling for credit and demographic covariates.

    Returns a dictionary containing:
      - 'robust_results': a simple namespace-like object exposing .params, .pvalues, .bse and .summary().as_text()
      - 'ame_female': average marginal effect of being female (difference in predicted probability at covariate means)
      - 'female_pvalue': p-value of the female coefficient under robust (HC3) covariance
      - 'summary_text': text summary of original (model-based) results with a note about HC3 robust inference

    Modeling approach:
      - Fit Logit(accept ~ female + controls)
      - Use robust (HC3) standard errors for inference
      - Compute an interpretable average marginal effect (AME) as predicted probability difference when female toggles 0->1 at means of controls
    """
    from scipy.special import expit

    # Ensure input has expected columns
    X_cols = [
        'female', 'mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std',
        'loan_to_value_std', 'housing_expense_ratio_std',
        'bad_history', 'black', 'married', 'self_employed'
    ]

    for c in X_cols + ['accept']:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe.")

    # Design matrix and outcome
    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['accept'].astype(int)

    # Fit Logit
    logit_model = sm.Logit(y, X)
    res = logit_model.fit(disp=0)

    # Attempt to obtain HC3 robust covariance via statsmodels robust results helper.
    try:
        robust_stats_res = res.get_robustcov_results(cov_type='HC3')
        cov_robust = robust_stats_res.cov_params()
        robust_bse = pd.Series(np.sqrt(np.diag(cov_robust)), index=res.params.index)
        robust_pvalues = pd.Series(robust_stats_res.pvalues, index=res.params.index)
    except Exception:
        # Fallback: compute HC3 robust covariance manually using design matrix and response residuals.
        # Using the HC3 formula: cov = (X'X)^{-1} X' diag((resid/(1-h))^2) X (X'X)^{-1}
        exog = res.model.exog  # design matrix used in fitting (includes constant)
        params_index = res.params.index

        # Residuals: prefer resid_response; otherwise compute y - mu
        try:
            resid = res.resid_response
        except Exception:
            # res.predict() returns predicted probabilities for Logit
            try:
                resid = res.model.endog - res.predict()
            except Exception:
                resid = res.model.endog - res.model.predict(res.params)

        # Compute (X'X)^{-1} robustly via pseudo-inverse for stability
        xtx = exog.T.dot(exog)
        try:
            inv_xtx = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            inv_xtx = np.linalg.pinv(xtx)

        # Leverages h = diag(X (X'X)^{-1} X')
        # Efficient computation without forming full hat matrix:
        h = np.sum(exog * (exog.dot(inv_xtx)), axis=1)

        # Avoid division by zero in (1 - h); if exactly 1, set to a tiny number to avoid inf
        one_minus_h = 1.0 - h
        one_minus_h[one_minus_h == 0.0] = np.finfo(float).eps

        u = resid / one_minus_h
        # S = X' diag(u^2) X
        S = exog.T.dot((u ** 2)[:, None] * exog)

        cov_robust = inv_xtx.dot(S).dot(inv_xtx)

        robust_bse = pd.Series(np.sqrt(np.abs(np.diag(cov_robust))), index=params_index)
        t_stats = res.params / robust_bse
        robust_pvalues = pd.Series(2.0 * norm.sf(np.abs(t_stats)), index=params_index)

    # Create a lightweight robust results wrapper exposing expected attributes
    summary_text = res.summary().as_text() + "\n\nNote: HC3 robust standard errors were computed separately and are used for p-values reported in the output."

    robust_res = SimpleNamespace(
        params=res.params,
        pvalues=robust_pvalues,
        bse=robust_bse,
        summary=lambda: SimpleNamespace(as_text=lambda: summary_text)
    )

    # Compute Average Marginal Effect (AME) for female at covariate means
    # Build baseline vector of means in the same order as model params (const first)
    means = df[X_cols].mean()
    base_vector = [1.0] + [means[c] for c in X_cols]

    # Index of female in base_vector (after const): position = 1 + index in X_cols
    female_pos = 1 + X_cols.index('female')

    # female = 1
    vec_female1 = np.array(base_vector, dtype=float)
    vec_female1[female_pos] = 1.0
    # female = 0
    vec_female0 = np.array(base_vector, dtype=float)
    vec_female0[female_pos] = 0.0

    # Use parameter estimates for prediction (robust affects SEs/p-values only)
    params_values = robust_res.params.values

    # Predicted probabilities using logistic link (expit)
    p1 = expit(np.dot(vec_female1, params_values))
    p0 = expit(np.dot(vec_female0, params_values))
    ame = float(p1 - p0)

    # Extract robust p-value for the female coefficient
    try:
        female_pvalue = float(robust_res.pvalues['female'])
    except Exception:
        female_idx = list(robust_res.params.index).index('female')
        female_pvalue = float(robust_res.pvalues.iloc[female_idx])

    # Prepare output
    results = {
        'robust_results': robust_res,
        'ame_female': ame,
        'female_pvalue': female_pvalue,
        'summary_text': robust_res.summary().as_text()
    }

    return results