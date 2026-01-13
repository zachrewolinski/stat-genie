from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sps

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/add_features_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Define columns needed for analysis (dependent, independent, and controls)
    required_cols = [
        'accept',        # dependent (1 = accepted, 0 = denied)
        'female',        # independent (1 = female, 0 = male)
        'black',
        'self_employed',
        'married',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'denied_PMI',
        'housing_expense_ratio',
        'occupation'
    ]

    # Keep only the relevant columns that are present in the input df
    present_cols = [c for c in required_cols if c in df.columns]
    df = df[present_cols].copy()

    # Coerce to numeric where appropriate
    for c in present_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any variable required for the model (original raw vars)
    df = df.dropna(subset=present_cols)

    # Ensure binary indicator columns are integer 0/1
    binary_cols = [c for c in ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI'] if c in df.columns]
    for c in binary_cols:
        # round then cast to int in case they are floats like 0.0/1.0
        df[c] = df[c].round().astype(int)

    # Standardize continuous/ordinal predictors to mean 0, sd 1 for numerical stability
    std_source_cols = [c for c in ['PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'mortgage_credit', 'consumer_credit', 'occupation'] if c in df.columns]
    for c in std_source_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        out_col = c + '_z'
        if std == 0 or np.isnan(std):
            df[out_col] = 0.0
        else:
            df[out_col] = (df[c] - mean) / std

    # Final dataframe returned contains all model-ready columns
    final_cols = []
    if 'accept' in df.columns:
        final_cols.append('accept')
    if 'female' in df.columns:
        final_cols.append('female')

    # Add binary controls in a fixed order if present
    for c in ['black', 'self_employed', 'married', 'bad_history', 'denied_PMI']:
        if c in df.columns:
            final_cols.append(c)

    # Add standardized numeric controls if present
    for c in ['PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'mortgage_credit_z', 'consumer_credit_z', 'occupation_z']:
        if c in df.columns:
            final_cols.append(c)

    # Return the dataframe with the chosen columns (model-ready)
    return df[final_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Prepare data (assumes df is output of transform)
    df = df.copy()

    # Dependent variable
    y = df['accept']

    # Independent and control variables to include in the model
    X_cols = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z',
        'mortgage_credit_z',
        'consumer_credit_z',
        'occupation_z'
    ]

    # Keep only columns that actually exist in the dataframe (robustness)
    X_cols = [c for c in X_cols if c in df.columns]

    # Add intercept
    X = sm.add_constant(df[X_cols], has_constant='add')

    # Fit binomial model using GLM (Binomial family). GLM is numerically stable and provides robust covariance support.
    try:
        glm_model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        robust_results = glm_model.get_robustcov_results(cov_type='HC3')
        return robust_results
    except Exception:
        # If GLM fails for some reason, fall back to Logit and compute robust covariance manually
        logit_model = sm.Logit(y, X).fit(disp=False)
        # Try to use built-in method if available
        if hasattr(logit_model, 'get_robustcov_results'):
            return logit_model.get_robustcov_results(cov_type='HC3')

        # Otherwise compute HC3-like sandwich covariance manually using score-based formula
        # Use model matrices available: X (DataFrame), y (Series)
        X_mat = np.asarray(X)
        y_arr = np.asarray(y)
        params = np.asarray(logit_model.params)

        # Predicted probabilities
        try:
            p = logit_model.predict(X)
            p = np.asarray(p)
        except Exception:
            linpred = X_mat @ params
            p = 1 / (1 + np.exp(-linpred))

        # Residuals (response residuals)
        resid = y_arr - p

        # Weight for Hessian (variance of Bernoulli)
        W = p * (1 - p)

        # Compute bread = inv(X' W X)
        XtWX = X_mat.T @ (W[:, None] * X_mat)
        try:
            bread = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(XtWX)

        # Compute meat = X' diag(resid^2) X
        meat = X_mat.T @ ((resid ** 2)[:, None] * X_mat)

        robust_cov = bread @ meat @ bread

        bse = np.sqrt(np.diag(robust_cov))
        z_vals = params / bse
        pvalues = 2 * (1 - sps.norm.cdf(np.abs(z_vals)))
        conf_int_arr = np.column_stack([params - 1.96 * bse, params + 1.96 * bse])

        class RobustResultsLike:
            def __init__(self, base_result, params, bse, pvalues, conf_int):
                self._base = base_result
                self.params = pd.Series(params, index=base_result.params.index)
                self.bse = pd.Series(bse, index=base_result.params.index)
                self.pvalues = pd.Series(pvalues, index=base_result.params.index)
                self._conf_int = pd.DataFrame(conf_int, index=base_result.params.index, columns=['2.5%', '97.5%'])

            def conf_int(self):
                return self._conf_int

            def summary(self):
                # Delegate to the original summary (will show non-robust SEs), but user can inspect params/bse/pvalues on this wrapper
                return self._base.summary()

        return RobustResultsLike(logit_model, params, bse, pvalues, conf_int_arr)