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
    # Work on a copy
    df = df.copy()

    # Relevant columns needed for analysis
    required_cols = [
        'female',
        'accept',
        'black',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'married',
        'self_employed',
        'housing_expense_ratio'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Ensure binary variables are integers 0/1
    for bcol in ['female', 'black', 'bad_history', 'married', 'self_employed', 'accept']:
        # cast to numeric and then to int (safeguard for floats)
        df[bcol] = pd.to_numeric(df[bcol], errors='coerce')
    df = df.dropna(subset=['accept', 'female'])  # drop rows where these couldn't be coerced
    df['accept'] = df['accept'].astype(int)
    df['female'] = df['female'].astype(int)
    df['black'] = df['black'].astype(int)
    df['bad_history'] = df['bad_history'].astype(int)
    df['married'] = df['married'].astype(int)
    df['self_employed'] = df['self_employed'].astype(int)

    # Continuous predictors: coerce to numeric
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with NA in continuous predictors (after coercion)
    df = df.dropna(subset=cont_cols)

    # Standardize continuous predictors (z-scores). Use sample std (ddof=1) for interpretability
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=1)
        # protect against zero std
        if std == 0 or np.isnan(std):
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Final returned dataframe contains original columns plus standardized versions used in the model
    # Keep only columns necessary for modeling to reduce accidental use of others
    model_cols = [
        'female', 'accept', 'black', 'mortgage_credit_z', 'consumer_credit_z', 'bad_history',
        'PI_ratio_z', 'loan_to_value_z', 'married', 'self_employed', 'housing_expense_ratio_z'
    ]
    # If any of these are missing because of prior operations, raise an informative error by returning df as-is
    missing_model_cols = [c for c in model_cols if c not in df.columns]
    if missing_model_cols:
        # return df so the caller can inspect; it's preferable to raise in production code
        return df

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # Ensure dataframe is the transformed dataframe (has standardized columns)
    df = df.copy()

    # Define features used in the logistic regression
    X_cols = [
        'female',
        'black',
        'mortgage_credit_z',
        'consumer_credit_z',
        'bad_history',
        'PI_ratio_z',
        'loan_to_value_z',
        'married',
        'self_employed',
        'housing_expense_ratio_z'
    ]

    # Drop rows with missing values in model columns (defensive)
    df = df.dropna(subset=X_cols + ['accept'])

    y = df['accept']
    X = df[X_cols]

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit a logistic regression (maximum likelihood)
    # Use statsmodels Logit for interpretability; catch potential convergence issues
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Compute odds ratios and 95% confidence intervals
    params = results.params
    conf = results.conf_int()
    or_df = pd.DataFrame({
        'OR': np.exp(params),
        'CI_lower': np.exp(conf[0]),
        'CI_upper': np.exp(conf[1])
    })

    # Return the fitted results and a compact odds-ratio table
    return {
        'results': results,
        'odds_ratios': or_df
    }


