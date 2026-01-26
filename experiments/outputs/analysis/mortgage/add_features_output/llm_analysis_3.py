from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/add_features_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling the effect of gender on mortgage acceptance.
    Steps:
    - Keep only rows with non-missing values for variables used in the model.
    - Cast binary flags to integers.
    - Ensure numeric columns are numeric.
    - Standardize continuous predictors (z-scores) to aid interpretation and model convergence.
    - Return a dataframe containing only the final columns used in the model.
    Final columns: ['female','accept','black','self_employed','married','bad_history','denied_PMI','occupation',
                    'housing_expense_ratio_z','PI_ratio_z','loan_to_value_z','mortgage_credit_z','consumer_credit_z']
    """
    df = df.copy()

    # Columns required for the analysis
    required = [
        'female', 'accept', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI', 'occupation'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required)

    # Ensure binary variables are integer 0/1
    binary_cols = ['female', 'accept', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for col in binary_cols:
        # convert to numeric then to int
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)

    # Ensure numeric columns are numeric
    numeric_cols = ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit', 'occupation']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # After coercion, drop any rows that became NA
    df = df.dropna(subset=numeric_cols)

    # Create standardized (z-scored) versions of continuous controls
    cont_to_z = ['housing_expense_ratio', 'PI_ratio', 'loan_to_value', 'mortgage_credit', 'consumer_credit']
    for col in cont_to_z:
        zcol = col + '_z'
        # use population std (ddof=0) for scaling to be explicit
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # fallback to avoid division by zero - create zeros
            df[zcol] = 0.0
        else:
            df[zcol] = (df[col] - mean) / std

    # Final columns used in the model
    final_cols = [
        'female', 'accept', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI', 'occupation',
        'housing_expense_ratio_z', 'PI_ratio_z', 'loan_to_value_z', 'mortgage_credit_z', 'consumer_credit_z'
    ]

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting the probability of mortgage acceptance (accept==1)
    from gender (female) while controlling for a set of applicant and loan characteristics.

    Returns a dictionary containing:
    - 'result': the fitted statsmodels LogitResults object
    - 'odds_ratios': pandas Series of exponentiated coefficients (odds ratios)
    - 'conf_int_odds': pandas DataFrame of exponentiated confidence intervals for odds ratios
    - 'model_summary': string summary (result.summary())
    """
    df = df.copy()

    # Dependent and independent variables
    y = df['accept']

    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI', 'occupation',
        'housing_expense_ratio_z', 'PI_ratio_z', 'loan_to_value_z', 'mortgage_credit_z', 'consumer_credit_z'
    ]

    X = df[X_cols]

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    try:
        result = logit.fit(disp=False)
    except Exception as e:
        # If convergence or perfect separation issues arise, try using a penalized approach (L2)
        # 'sm.Logit' supports method='newton' with regularization via 'fit_regularized'
        result = logit.fit_regularized(method='lbfgs')

    # Compute odds ratios and 95% CI for odds ratios
    params = result.params
    conf = result.conf_int()
    conf.columns = ['2.5%', '97.5%']

    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    return {
        'result': result,
        'odds_ratios': odds_ratios,
        'conf_int_odds': conf_odds,
        'model_summary': result.summary().as_text()
    }


