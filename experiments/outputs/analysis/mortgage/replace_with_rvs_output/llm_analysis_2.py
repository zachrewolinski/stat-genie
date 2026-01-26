from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the Boston mortgage dataset for logistic regression.

    Produces standardized continuous controls and keeps the binary controls.
    Returns a dataframe that contains all columns used in the model.
    """
    df = df.copy()

    # Columns required for the analysis
    required_cols = [
        'accept',         # dependent variable
        'female',         # independent variable
        'black',
        'married',
        'self_employed',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'housing_expense_ratio',
        'mortgage_credit',
        'consumer_credit'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are numeric 0/1
    binary_cols = ['accept', 'female', 'black', 'married', 'self_employed', 'bad_history']
    for c in binary_cols:
        # coerce to numeric and then to 0/1 if possible
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=binary_cols)

    # Standardize continuous/ordinal controls for interpretability
    std_cols = ['PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'mortgage_credit', 'consumer_credit']
    for c in std_cols:
        col_vals = pd.to_numeric(df[c], errors='coerce')
        mean = col_vals.mean()
        std = col_vals.std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, create zero column (will be dropped by model fitting if problematic)
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (col_vals - mean) / std

    # Final set of columns we will use in modeling
    final_cols = [
        'accept', 'female', 'black', 'married', 'self_employed', 'bad_history',
        'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'mortgage_credit_z', 'consumer_credit_z'
    ]

    # Keep only final columns (and drop rows with any remaining NaNs)
    df_final = df[final_cols].dropna()

    # Ensure dtypes are numeric
    df_final = df_final.astype({c: float for c in df_final.columns})

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression to estimate the effect of gender on mortgage acceptance,
    controlling for applicant financial and demographic characteristics.

    Returns the fitted statsmodels Logit result object and a small summary dictionary
    with odds ratios and 95% confidence intervals.
    """
    import statsmodels.api as sm
    import numpy as np

    # Prepare design matrix and response
    X_cols = [
        'female', 'black', 'married', 'self_employed', 'bad_history',
        'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'mortgage_credit_z', 'consumer_credit_z'
    ]

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (maximum likelihood)
    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    # Compute odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    or_vals = np.exp(params)
    or_lower = np.exp(conf[0])
    or_upper = np.exp(conf[1])

    summary_dict = {
        'params': params,
        'odds_ratios': or_vals,
        'or_ci_lower': or_lower,
        'or_ci_upper': or_upper,
        'pvalues': result.pvalues,
        'nobs': int(result.nobs)
    }

    return result, summary_dict


