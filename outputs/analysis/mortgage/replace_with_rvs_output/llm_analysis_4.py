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
    Prepare data for modeling the effect of gender on mortgage acceptance.

    Steps:
    - Make a copy of the input dataframe to avoid side effects.
    - Ensure binary columns are numeric (0/1) and the dependent 'accept' is numeric.
    - Create standardized (z-scored) versions of continuous controls to aid model convergence and interpretability.
    - Drop rows with missing values in any columns used in the model.

    Returns a dataframe containing all columns listed in the conceptual variables.
    """
    df = df.copy()

    # Ensure dependent and main independent variable exist and are numeric
    if 'accept' not in df.columns:
        raise KeyError("Input dataframe must contain 'accept' column as the dependent variable.")
    if 'female' not in df.columns:
        raise KeyError("Input dataframe must contain 'female' column as the primary independent variable.")

    # Convert to numeric if necessary
    df['accept'] = pd.to_numeric(df['accept'], errors='coerce')
    df['female'] = pd.to_numeric(df['female'], errors='coerce')

    # Binary controls: ensure numeric
    binary_cols = ['black', 'bad_history', 'married', 'self_employed', 'denied_PMI']
    for c in binary_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # If a binary control is missing from the dataset, create a column of NA so dropna will remove rows
            df[c] = np.nan

    # Continuous controls to standardize (if missing create NA column)
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # Create z-scored versions for continuous controls; guard against zero std
    for c in cont_cols:
        zname = 'z_' + c
        if df[c].dropna().shape[0] == 0:
            df[zname] = np.nan
        else:
            mean = df[c].mean()
            std = df[c].std()
            if std == 0 or np.isnan(std):
                # if zero variance, subtract mean only (results in zeros or NaN)
                df[zname] = df[c] - mean
            else:
                df[zname] = (df[c] - mean) / std

    # Select only the columns that will be used in the model
    model_cols = [
        'accept',
        'female',
        'black',
        'bad_history',
        'married',
        'self_employed',
        'denied_PMI',
        'z_mortgage_credit',
        'z_consumer_credit',
        'z_PI_ratio',
        'z_loan_to_value',
        'z_housing_expense_ratio'
    ]

    # If any expected z_ columns were not created above because the original column was missing, ensure they exist
    for col in model_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows with missing values in any model column (listwise deletion)
    df_model = df[model_cols].dropna(axis=0, how='any').copy()

    # Ensure dtypes are numeric and integers for binary indicators
    df_model['accept'] = df_model['accept'].astype(int)
    df_model['female'] = df_model['female'].astype(int)
    for b in ['black', 'bad_history', 'married', 'self_employed', 'denied_PMI']:
        df_model[b] = df_model[b].astype(int)

    # Return the prepared dataframe
    return df_model


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binary outcome) predicting acceptance of a mortgage application.

    Model: logit( P(accept=1) ) = alpha + beta_female * female + gamma' * controls

    Controls included (as prepared by transform): black, bad_history, married, self_employed, denied_PMI,
    z_mortgage_credit, z_consumer_credit, z_PI_ratio, z_loan_to_value, z_housing_expense_ratio.

    Returns the fitted statsmodels result object (LogitResults).
    """
    # Required columns
    X_cols = [
        'female',
        'black',
        'bad_history',
        'married',
        'self_employed',
        'denied_PMI',
        'z_mortgage_credit',
        'z_consumer_credit',
        'z_PI_ratio',
        'z_loan_to_value',
        'z_housing_expense_ratio'
    ]
    y_col = 'accept'

    # Ensure required columns exist
    missing = [c for c in X_cols + [y_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe is missing required columns for modeling: {missing}")

    X = df[X_cols].astype(float)
    y = df[y_col].astype(int)

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (use robust method by default). Suppress convergence output.
    model = sm.Logit(y, X)
    try:
        results = model.fit(disp=False)
    except Exception as e:
        # If default fit fails, try a stronger optimizer
        results = model.fit(method='bfgs', disp=False)

    # Optionally compute average marginal effect of 'female'
    # (user can call results.get_margeff() externally if needed). We return the fitted results object.
    return results


