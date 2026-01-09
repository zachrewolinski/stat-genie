from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/examples/mortgage/analysis5_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw HMDA-style dataframe into the final analysis dataframe.

    Steps:
    - Work on a copy.
    - Drop rows missing the dependent variable (accept) or the key independent variable (female).
    - Drop rows missing core control variables necessary for a fair comparison.
    - Coerce core binary vars to integers (0/1).
    - Standardize continuous numeric controls (z-scores) and store them as new columns with _z suffix.
    - Return the dataframe including the original columns plus the derived standardized columns.
    """
    df = df.copy()

    # Columns we require for analysis
    required_cols = [
        'accept', 'female',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value',
        'bad_history', 'self_employed', 'married', 'black', 'housing_expense_ratio', 'denied_PMI'
    ]

    # Drop rows that are missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are integers (0/1)
    for col in ['accept', 'female', 'bad_history', 'self_employed', 'married', 'black', 'denied_PMI']:
        # some datasets might have floats like 0.0/1.0; cast to int
        df[col] = df[col].astype(int)

    # Standardize continuous numeric controls (z-scores). Use population std (ddof=0) to match many ML workflows.
    cont_to_z = ['housing_expense_ratio', 'PI_ratio', 'mortgage_credit', 'consumer_credit', 'loan_to_value']
    for col in cont_to_z:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # If std is zero (unlikely), create zeros to avoid divide-by-zero
        if std == 0 or np.isnan(std):
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = (df[col] - mean) / std

    # Final check: keep only rows that still have no missing values in the derived columns and required binaries
    final_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
                  'housing_expense_ratio_z', 'PI_ratio_z', 'mortgage_credit_z', 'consumer_credit_z', 'loan_to_value_z']
    df = df.dropna(subset=final_cols)

    # Return the dataframe with both original and derived variables. The modeling function will select needed cols.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting mortgage acceptance (accept) from female and controls.

    Returns a dict with the fitted model (statsmodels results) and a table of odds ratios with 95% CIs and p-values.
    """
    # Select columns used in the model (must match the names produced in transform)
    X_cols = [
        'female',
        'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'housing_expense_ratio_z', 'PI_ratio_z', 'mortgage_credit_z', 'consumer_credit_z', 'loan_to_value_z'
    ]

    # Ensure all columns present
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    X = df[X_cols]
    y = df['accept']

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    # Use try/except to surface convergence problems cleanly
    try:
        logit_model = sm.Logit(y, X)
        res = logit_model.fit(disp=False, maxiter=200)
    except Exception as e:
        # If Logit fails (e.g., perfect separation), raise with context
        raise RuntimeError(f"Logit model failed to fit: {e}")

    # Compute odds ratios and 95% CI
    params = res.params
    conf = res.conf_int()
    or_series = np.exp(params)
    ci_lower = np.exp(conf[0])
    ci_upper = np.exp(conf[1])

    or_table = pd.DataFrame({
        'OR': or_series,
        'CI_lower': ci_lower,
        'CI_upper': ci_upper,
        'p_value': res.pvalues
    })

    # Return results object and the OR table for reporting
    results = {
        'model_results': res,
        'odds_ratios': or_table
    }
    return results


