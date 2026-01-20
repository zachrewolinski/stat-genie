from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/add_features_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Columns required for the analysis
    required_cols = [
        'accept',  # dependent variable
        'female',  # independent variable
        'black', 'self_employed', 'married', 'bad_history',  # binary controls
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'occupation'
    ]

    # Keep only required columns (drop extras)
    df = df.loc[:, [c for c in required_cols if c in df.columns]]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Ensure binary columns are integers (0/1)
    for bcol in ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history']:
        if bcol in df.columns:
            # Coerce to numeric then to int (safe conversion)
            df[bcol] = pd.to_numeric(df[bcol], errors='coerce').astype(int)

    # Ensure continuous columns are numeric
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'occupation']
    for c in cont_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop any rows that became NaN after coercion
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Standardize continuous predictors (z-score) and store with _z suffix
    # Use population std (ddof=0) for consistency
    for c in cont_cols:
        if c in df.columns:
            zcol = f"{c}_z"
            mean = df[c].mean()
            std = df[c].std(ddof=0)
            # If std is zero (constant), create zeros to avoid division by zero
            if std == 0 or np.isnan(std):
                df[zcol] = 0.0
            else:
                df[zcol] = (df[c] - mean) / std

    # Final dataframe contains the dependent variable, the independent variable, and standardized controls
    final_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'occupation_z'
    ]

    # Keep only columns that exist (in case some expected cols were not in the original df)
    final_cols = [c for c in final_cols if c in df.columns]
    df = df.loc[:, final_cols]

    # Return transformed dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build logistic regression predicting acceptance (approval) from gender and controls
    import statsmodels.api as sm

    # Ensure the dataframe contains the columns we expect
    predictors = [
        'female', 'black', 'self_employed', 'married', 'bad_history',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'occupation_z'
    ]
    predictors = [p for p in predictors if p in df.columns]

    # Dependent variable
    if 'accept' not in df.columns:
        raise ValueError("Transformed dataframe must contain 'accept' column as the dependent variable.")

    y = df['accept']
    X = df[predictors]

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Return the fitted results object (user can call .summary(), .params, etc.)
    return results


