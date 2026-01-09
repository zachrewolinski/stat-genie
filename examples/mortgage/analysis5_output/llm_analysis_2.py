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
    Transform the raw HMDA-style mortgage dataset to produce the columns used in modeling.

    Steps:
    - Make a copy of the dataframe.
    - Drop rows missing the outcome (accept) or the key independent variable (female) or essential controls.
    - Coerce binary indicator columns to 0/1 ints.
    - Standardize continuous controls (z-score) and add new _z columns.
    - Create an interaction term female_black to test whether the gender effect differs by race.

    Returns the dataframe with all columns referenced in the conceptual variables.
    """
    df = df.copy()

    # Columns required for analysis
    required_cols = [
        'accept',
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'mortgage_credit',
        'consumer_credit',
        'PI_ratio',
        'loan_to_value',
        'housing_expense_ratio'
    ]

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are numeric 0/1
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for col in binary_cols:
        # Coerce to numeric and then to int (if values are 0/1 floats)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].astype(int)

    # Standardize continuous predictors: (x - mean) / std. Create new _z columns used in the model.
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in cont_cols:
        # convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # If std is zero (unlikely) avoid division by zero
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Interaction: female * black
    df['female_black'] = df['female'] * df['black']

    # Keep only the columns we will use in modeling to avoid accidental use of other columns
    keep_cols = [
        'accept',
        'female',
        'black',
        'female_black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z'
    ]

    # If any standardized columns are missing because original cont cols were constant or absent, ensure they exist
    for col in ['mortgage_credit_z','consumer_credit_z','PI_ratio_z','loan_to_value_z','housing_expense_ratio_z']:
        if col not in df.columns:
            df[col] = 0.0

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (logit) predicting mortgage acceptance from applicant gender,
    race (Black), the female*black interaction, and a set of applicant financial and demographic controls.

    Returns the fitted statsmodels LogitResults object.
    """
    import statsmodels.api as sm

    # Define outcome and predictors used in the conceptual model
    y = df['accept'].astype(float)

    X_cols = [
        'female',
        'black',
        'female_black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z'
    ]

    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression model using maximum likelihood
    logit_model = sm.Logit(y, X)
    try:
        results = logit_model.fit(disp=False)
    except Exception:
        # If the model has convergence issues, try a GLM with binomial family (more numerically stable in some cases)
        results = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    return results


