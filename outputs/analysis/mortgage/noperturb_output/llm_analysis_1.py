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
    # Work on a copy to avoid mutating original
    df = df.copy()

    # Ensure binary outcome and key indicators are numeric
    # (If they are stored as floats/ints already this will be a no-op)
    df['accept'] = pd.to_numeric(df['accept'], errors='coerce')
    df['female'] = pd.to_numeric(df['female'], errors='coerce')
    df['black'] = pd.to_numeric(df['black'], errors='coerce')
    df['bad_history'] = pd.to_numeric(df['bad_history'], errors='coerce')
    df['married'] = pd.to_numeric(df['married'], errors='coerce')
    df['self_employed'] = pd.to_numeric(df['self_employed'], errors='coerce')
    df['denied_PMI'] = pd.to_numeric(df['denied_PMI'], errors='coerce')

    # Numeric controls
    df['mortgage_credit'] = pd.to_numeric(df['mortgage_credit'], errors='coerce')
    df['consumer_credit'] = pd.to_numeric(df['consumer_credit'], errors='coerce')
    df['PI_ratio'] = pd.to_numeric(df['PI_ratio'], errors='coerce')
    df['loan_to_value'] = pd.to_numeric(df['loan_to_value'], errors='coerce')
    df['housing_expense_ratio'] = pd.to_numeric(df['housing_expense_ratio'], errors='coerce')

    # Drop rows with missing values in any variables used in the model
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'bad_history', 'PI_ratio', 'loan_to_value', 'married', 'self_employed',
        'housing_expense_ratio', 'denied_PMI'
    ]
    df = df.dropna(subset=required_cols)

    # Standardize continuous numeric controls (z-score). Use sample std (ddof=0 or ddof=1 both acceptable; using ddof=0 here).
    for col in ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # If std is 0 (constant column), create zeros to avoid division by zero
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Finalize column names used in the model
    # mortgage_credit_z, consumer_credit_z, PI_ratio_z, loan_to_value_z, housing_expense_ratio_z
    # Keep original binary variables as-is

    # Return only the columns necessary for modeling to keep the DataFrame focused
    model_cols = [
        'accept', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
        'bad_history', 'PI_ratio_z', 'loan_to_value_z', 'married', 'self_employed',
        'housing_expense_ratio_z', 'denied_PMI'
    ]
    # If any of the z columns aren't present for some reason, raise an informative error
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after transform: {missing}")

    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # df is expected to be the transformed dataframe returned by transform()
    # Prepare design matrix and outcome
    y = df['accept']

    X = df[[
        'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
        'bad_history', 'PI_ratio_z', 'loan_to_value_z', 'married',
        'self_employed', 'housing_expense_ratio_z', 'denied_PMI'
    ]]

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    # Use try/except to surface convergence issues clearly
    try:
        logit_model = sm.Logit(y, X)
        results = logit_model.fit(disp=False, method='lbfgs')
    except Exception as e:
        # If Logit fails (perfect separation or convergence), try GLM with binomial family as a fallback
        try:
            glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
            results = glm_binom.fit()
        except Exception:
            # Re-raise the original exception if fallback also fails
            raise e

    # Return the fitted results object (has params, summary(), pvalues, conf_int(), etc.)
    return results


