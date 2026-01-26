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
    Prepare data for modeling the effect of gender on mortgage acceptance.
    Steps:
    - Keep only relevant columns.
    - Coerce types and handle missing/infinite values.
    - Drop rows with missing DV (accept) or IV (female) or key controls.
    - Standardize continuous controls and create *_s columns used in the model.
    - Return dataframe that contains the exact column names referenced in the model.
    """
    df = df.copy()

    # Columns we intend to use
    required_cols = [
        'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed',
        'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'denied_PMI'
    ]

    # Ensure columns exist (if dataset has slightly different naming, this will raise a clear error)
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Subset to required columns first (keep others but we'll work with this subset)
    df = df[required_cols].copy()

    # Coerce binary indicators to numeric (0/1)
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for col in binary_cols:
        # convert to numeric and coerce invalid to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # allow only 0/1; if other values present, leave as is (they will be dropped by dropna)

    # Continuous controls: ensure numeric
    cont_cols = ['housing_expense_ratio', 'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value']
    for col in cont_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # replace infinities with NaN
        df.loc[np.isinf(df[col]), col] = np.nan

    # Drop rows with missing DV or IV or any control used in model
    df = df.dropna(subset=['accept', 'female'] + binary_cols + cont_cols)

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for stability.
    for col in cont_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If zero variance, create zero column to avoid divide-by-zero
            df[col + '_s'] = 0.0
        else:
            df[col + '_s'] = (df[col] - mean) / std

    # Create the final dataframe keeping only columns that will be used in the model
    final_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'mortgage_credit_s', 'consumer_credit_s', 'housing_expense_ratio_s', 'PI_ratio_s', 'loan_to_value_s'
    ]

    # Ensure final columns exist (they should after standardization)
    missing_final = [c for c in final_cols if c not in df.columns]
    if len(missing_final) > 0:
        raise KeyError(f"Transformation failed to produce required final columns: {missing_final}")

    # Reorder and return
    df = df[final_cols].reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (binomial) predicting acceptance (accept) from applicant gender (female)
    controlling for creditworthiness and demographic/loan characteristics.

    Returns a dictionary containing the fitted model, summary text, and a table of odds ratios with 95% CIs.
    """
    # Ensure input is the transformed dataframe with required columns
    required_model_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'mortgage_credit_s', 'consumer_credit_s', 'housing_expense_ratio_s', 'PI_ratio_s', 'loan_to_value_s'
    ]
    missing = [c for c in required_model_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Outcome and predictors
    y = df['accept'].astype(float)
    X = df[[
        'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'mortgage_credit_s', 'consumer_credit_s', 'housing_expense_ratio_s', 'PI_ratio_s', 'loan_to_value_s'
    ]].astype(float)

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression using statsmodels GLM with binomial family (stable and gives summary)
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Prepare odds ratios and 95% CI
    params = res.params
    conf = res.conf_int()
    or_vals = np.exp(params)
    conf_exp = np.exp(conf)
    or_table = pd.DataFrame({
        'coef': params,
        'odds_ratio': or_vals,
        'ci_lower': conf_exp[0],
        'ci_upper': conf_exp[1]
    })

    output = {
        'model_result': res,
        'summary_text': res.summary().as_text(),
        'odds_ratio_table': or_table
    }
    return output


