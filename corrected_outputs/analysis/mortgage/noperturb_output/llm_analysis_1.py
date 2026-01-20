from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/noperturb_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling:
    - Keep a copy of the dataframe.
    - Drop rows with missing values in the dependent, independent, and control columns listed in the conceptual model.
    - Ensure binary indicator columns are integer type.
    - Standardize continuous ratio variables (PI_ratio, loan_to_value, housing_expense_ratio) and add them as new columns prefixed with 'std_'.

    Returns the transformed dataframe that contains exactly the columns referenced in the model.
    """
    df = df.copy()

    # Columns required for the analysis
    required_cols = [
        'accept',
        'female',
        'black',
        'married',
        'self_employed',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'housing_expense_ratio',
        'denied_PMI'
    ]

    # Drop rows missing any required value
    df = df.dropna(subset=required_cols)

    # Coerce binary indicators to integer (0/1)
    binary_cols = ['accept', 'female', 'black', 'married', 'self_employed', 'bad_history', 'denied_PMI']
    for col in binary_cols:
        # Some columns might already be floats with 0.0/1.0; cast to int
        df[col] = df[col].astype(int)

    # Ensure credit score columns are numeric (they are ordinal but kept as numeric predictors)
    df['mortgage_credit'] = pd.to_numeric(df['mortgage_credit'], errors='coerce')
    df['consumer_credit'] = pd.to_numeric(df['consumer_credit'], errors='coerce')

    # If conversion introduced NaNs, drop them
    df = df.dropna(subset=['mortgage_credit', 'consumer_credit'])

    # Standardize continuous ratio predictors and produce new columns used by the model
    cont_cols = ['PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in cont_cols:
        std_name = 'std_' + col
        # Use sample std (ddof=1) for standardization
        mean = df[col].mean()
        std = df[col].std()
        # If std is 0 (unlikely), set to 1 to avoid division by zero
        if pd.isna(std) or std == 0:
            df[std_name] = df[col] - mean
        else:
            df[std_name] = (df[col] - mean) / std

    # Final dataframe contains all columns used in the model
    final_cols = [
        'accept',
        'female',
        'black',
        'married',
        'self_employed',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'std_PI_ratio',
        'std_loan_to_value',
        'std_housing_expense_ratio',
        'denied_PMI'
    ]

    # Keep only final columns (but return full df could be useful; we return full df with these columns present)
    # Here we ensure the dataframe contains those columns and drop rows if any of them are missing
    df = df.dropna(subset=final_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (logit) predicting mortgage acceptance (accept) from applicant gender (female)
    while adjusting for the controls specified in the conceptual model. Returns the fitted results object,
    odds ratios with confidence intervals, and average marginal effect summary (if available).

    Model specification (in words):
      logit(P(accept=1)) = intercept + beta_female * female + sum_k beta_k * control_k

    Controls included: black, married, self_employed, bad_history, mortgage_credit, consumer_credit,
    std_PI_ratio, std_loan_to_value, std_housing_expense_ratio, denied_PMI
    """
    # Make a local copy
    df = df.copy()

    # Define regressors (independent variable + controls) in the exact column names used in transform
    X_cols = [
        'female',
        'black',
        'married',
        'self_employed',
        'bad_history',
        'mortgage_credit',
        'consumer_credit',
        'std_PI_ratio',
        'std_loan_to_value',
        'std_housing_expense_ratio',
        'denied_PMI'
    ]

    # Ensure the columns exist
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"The dataframe is missing required columns for the model: {missing}")

    X = df[X_cols]
    # Add constant (intercept)
    X = sm.add_constant(X, has_constant='add')

    # Dependent variable
    y = df['accept']

    # Fit logistic regression
    logit_model = sm.Logit(y, X)
    # Use default optimizer, suppress output
    result = logit_model.fit(disp=False)

    # Compute odds ratios and exponentiated confidence intervals
    odds_ratios = np.exp(result.params)

    # Obtain confidence intervals robustly
    conf = result.conf_int()
    # conf is a DataFrame with two columns (lower, upper); use iloc to extract columns
    conf_lower = conf.iloc[:, 0]
    conf_upper = conf.iloc[:, 1]

    conf_int_exp = pd.DataFrame({
        'OR': odds_ratios,
        'CI_lower': np.exp(conf_lower),
        'CI_upper': np.exp(conf_upper)
    })

    # Attempt to compute average marginal effects (AME); if it fails, return None for that item
    try:
        margeff = result.get_margeff()
        # summary() returns a Summary object; convert to string for easy display/storage
        margeff_summary = str(margeff.summary())
        # Also expose the marginal effects table as a DataFrame if possible
        try:
            margeff_table = margeff.summary_frame()
        except Exception:
            margeff_table = None
    except Exception:
        margeff_summary = None
        margeff_table = None

    results = {
        'model_result': result,                # statsmodels result object
        'odds_ratios_and_CI': conf_int_exp,   # DataFrame with OR and exponentiated CI
        'marginal_effects_summary': margeff_summary,
        'marginal_effects_table': margeff_table
    }

    return results