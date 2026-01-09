from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/mortgage/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe. The function:
      - Keeps only rows with non-missing values for the variables used in the model.
      - Ensures binary columns are integers (0/1).
      - Standardizes continuous/ordinal predictors to z-scores to aid interpretation and numerical stability.
    The returned dataframe contains the original binary indicators plus the *_z standardized columns used in the model.
    """
    df = df.copy()

    # Columns required for the analysis
    required_cols = [
        'female', 'accept', 'black', 'mortgage_credit', 'consumer_credit',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'bad_history', 'married', 'self_employed', 'denied_PMI'
    ]

    # Coerce to numeric where possible (will convert problematic strings to NaN)
    for col in required_cols:
        # keep original types if already numeric, otherwise coerce
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing data in any required column
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Ensure binary indicator columns are integer 0/1
    binary_cols = ['female', 'accept', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']
    for col in binary_cols:
        # round then convert to int to handle floats like 0.0/1.0
        df[col] = df[col].round().astype(int)

    # Create standardized (z-scored) versions of continuous / ordinal predictors
    z_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in z_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # if constant (unexpected), create zeros
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Final check: keep only the columns we will use in the model + originals for traceability
    # (model uses standardized columns where applicable)
    keep_cols = [
        'female', 'accept', 'black', 'mortgage_credit_z', 'consumer_credit_z',
        'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'bad_history', 'married', 'self_employed', 'denied_PMI'
    ]

    # Confirm all keep_cols exist; if some z-columns are missing due to earlier naming, raise error
    missing = [c for c in keep_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing expected transformed columns: {missing}")

    # Return dataframe containing at least the variables needed for modeling
    return df[keep_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (binary outcome: accept) to estimate the effect of gender (female)
    on the probability of mortgage application acceptance adjusting for observed controls.

    Returns a dictionary with the fitted statsmodels result object and a table of odds ratios
    with 95% confidence intervals and p-values.
    """
    # Features included in the model (must match the transformed dataframe column names)
    features = [
        'female', 'black', 'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z',
        'loan_to_value_z', 'housing_expense_ratio_z', 'bad_history', 'married',
        'self_employed', 'denied_PMI'
    ]

    # Ensure all features are present
    missing = [f for f in features + ['accept'] if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    X = df[features]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression using statsmodels Logit
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Compute odds ratios and 95% confidence intervals
    params = results.params
    conf = results.conf_int()
    or_table = pd.DataFrame({
        'OR': np.exp(params),
        'CI_lower': np.exp(conf[0]),
        'CI_upper': np.exp(conf[1]),
        'p_value': results.pvalues
    })

    # Print summary for quick inspection (can be removed if used in non-interactive pipelines)
    print(results.summary())
    print('\nOdds ratios (exp(coef)) with 95% CI:\n')
    print(or_table)

    return {'model_result': results, 'odds_ratios': or_table}


