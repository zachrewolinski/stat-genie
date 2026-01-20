from typing import Any
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Boston mortgage dataset for modeling the effect of gender on mortgage acceptance.

    Steps:
    - Keep only columns required for analysis.
    - Ensure numeric types and drop rows with missing values in required columns.
    - Create z-scored (standardized) versions of continuous/ordinal controls to aid interpretation and model stability.

    Returns the transformed dataframe including original binary controls and standardized continuous controls.
    """
    df = df.copy()

    # Columns required for the analysis (final dataframe must contain these names)
    required_cols = [
        'accept',        # dependent variable
        'female',        # independent variable
        'black',
        'married',
        'self_employed',
        'bad_history',
        'denied_PMI',
        'PI_ratio',
        'loan_to_value',
        'housing_expense_ratio',
        'mortgage_credit',
        'consumer_credit'
    ]

    # Keep only these columns if they exist
    existing = [c for c in required_cols if c in df.columns]
    df = df[existing].copy()

    # Coerce to numeric where appropriate (safeguard against strings)
    for col in existing:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in any required column
    df = df.dropna(subset=existing)

    # Standardize continuous / ordinal predictors to z-scores and create *_z columns
    to_z = [
        'PI_ratio',
        'loan_to_value',
        'housing_expense_ratio',
        'mortgage_credit',
        'consumer_credit'
    ]

    for col in to_z:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            # If std is zero (constant column), create z column filled with 0.0
            if std == 0 or np.isnan(std):
                df[col + '_z'] = 0.0
            else:
                df[col + '_z'] = (df[col] - mean) / std

    # Final sanity checks: ensure dependent and independent are binary (0/1)
    # If 'accept' is not strictly 0/1, coerce positive values to 1 and others to 0
    if 'accept' in df.columns:
        unique_accept = pd.unique(df['accept'])
        if not set(unique_accept).issubset({0, 1}):
            df['accept'] = df['accept'].apply(lambda x: 1 if x and x != 0 else 0)
    # Ensure integer dtype for binary columns if present
    binary_cols = ['accept', 'female', 'black', 'married', 'self_employed', 'bad_history', 'denied_PMI']
    for bcol in binary_cols:
        if bcol in df.columns:
            # Map non-binary truthy values to 1, falsy/0 to 0, then cast to int
            unique_vals = pd.unique(df[bcol])
            if not set(unique_vals).issubset({0, 1}):
                df[bcol] = df[bcol].apply(lambda x: 1 if x and x != 0 else 0)
            # Now cast safely
            df[bcol] = df[bcol].astype(int)

    # Ensure all final required *_z columns exist in the final dataframe (create if missing filled with 0)
    for col in to_z:
        zcol = col + '_z'
        if zcol not in df.columns:
            df[zcol] = 0.0

    # Reset index to keep dataframe tidy
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binary outcome) predicting mortgage acceptance from gender
    controlling for applicant and loan characteristics. Returns a results object with robust SEs.

    Model specification:
    accept ~ female + black + married + self_employed + bad_history + denied_PMI
             + PI_ratio_z + loan_to_value_z + housing_expense_ratio_z
             + mortgage_credit_z + consumer_credit_z

    Uses statsmodels Logit with heteroskedasticity-robust covariance (HC1).
    """
    # Columns used in the model (must match the transformed dataframe)
    X_cols = [
        'female',
        'black',
        'married',
        'self_employed',
        'bad_history',
        'denied_PMI',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z',
        'mortgage_credit_z',
        'consumer_credit_z'
    ]

    # Confirm columns exist
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare X and y
    X = df[X_cols].copy()
    y = df['accept'].copy()

    # Ensure numeric types
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop rows with NA in X or y (shouldn't normally happen after transform)
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Detect and drop any columns with zero variance (constant columns cause singular matrix)
    const_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    # Do not remove the constant column named 'const' if it's the only column left;
    # but if it's constant because all predictors were constant, model cannot be estimated.
    if const_cols:
        # Prefer to keep 'const' but drop other constant predictors
        to_drop = [c for c in const_cols if c != 'const']
        if to_drop:
            warnings.warn(f"Dropping constant predictor columns to avoid singular matrix: {to_drop}")
            X = X.drop(columns=to_drop)

    if X.shape[1] <= 1:
        # Only constant column left (or no predictors) -> cannot fit model
        raise ValueError("No variation in predictors after removing constant columns; cannot fit model.")

    # Try fitting Logit, with fallback to GLM if linear algebra issues occur
    try:
        logit = sm.Logit(y, X)
        res = logit.fit(disp=False)
    except np.linalg.LinAlgError:
        warnings.warn("Logit fit encountered a linear algebra error (singular matrix). Falling back to GLM Binomial.")
        try:
            glm = sm.GLM(y, X, family=sm.families.Binomial())
            res = glm.fit()
        except Exception as e:
            raise RuntimeError("Both Logit and GLM fitting failed.") from e
    except Exception as e:
        # Re-raise other exceptions with context
        raise

    # Convert to robust covariance (HC1) for heteroskedasticity-robust SEs
    try:
        res_robust = res.get_robustcov_results(cov_type='HC1')
    except Exception:
        # If robust covariance computation fails for this results object, return the original result
        warnings.warn("Could not compute robust covariance; returning plain fit results.")
        res_robust = res

    # Print a brief summary (caller can print full summary if desired)
    try:
        print(res_robust.summary())
    except Exception:
        pass

    return res_robust