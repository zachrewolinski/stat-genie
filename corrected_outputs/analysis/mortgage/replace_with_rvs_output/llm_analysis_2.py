from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw HMDA-derived dataset to a modeling dataframe.

    Outputs (columns used in the model):
      - Accept: binary outcome (1 = accepted, 0 = denied)
      - Female: binary gender indicator (1 = female, 0 = male)
      - Black: binary race indicator (1 = Black, 0 otherwise)
      - Female_Black: product Female * Black (interaction)
      - mortgage_credit_z, consumer_credit_z, PI_ratio_z, loan_to_value_z, housing_expense_ratio_z: z-scored continuous controls
      - SelfEmployed, Married, BadHistory: binary controls

    The function drops rows with missing values in the required fields and returns the dataframe with the new columns appended.
    """
    df = df.copy()

    # Required raw columns used to build the modeling dataset
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'bad_history', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'self_employed', 'married'
    ]

    # Ensure numeric where expected
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing any of the required raw variables
    df = df.dropna(subset=required_cols)

    # Create the dependent and independent variables with clear names used in the model
    df['Accept'] = df['accept'].astype(int)
    df['Female'] = df['female'].astype(int)
    df['Black'] = df['black'].astype(int)
    df['SelfEmployed'] = df['self_employed'].astype(int)
    df['Married'] = df['married'].astype(int)
    df['BadHistory'] = df['bad_history'].astype(int)

    # Continuous variables: compute z-scores (mean 0, sd 1) to aid interpretation and numeric stability
    # Use population std (ddof=0) for standardization; falls back to pandas default behavior
    cont_map = {
        'mortgage_credit': 'mortgage_credit_z',
        'consumer_credit': 'consumer_credit_z',
        'PI_ratio': 'PI_ratio_z',
        'loan_to_value': 'loan_to_value_z',
        'housing_expense_ratio': 'housing_expense_ratio_z'
    }

    for raw_col, z_col in cont_map.items():
        mean = df[raw_col].mean()
        std = df[raw_col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If there's no variation, create a zero column to avoid divide-by-zero
            df[z_col] = 0.0
        else:
            df[z_col] = (df[raw_col] - mean) / std

    # Interaction to test whether the gender effect differs by race
    df['Female_Black'] = df['Female'] * df['Black']

    # Final drop in case any transformation introduced NA
    model_cols = [
        'Accept', 'Female', 'Black', 'Female_Black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z',
        'loan_to_value_z', 'housing_expense_ratio_z',
        'SelfEmployed', 'Married', 'BadHistory'
    ]

    df = df.dropna(subset=model_cols)

    # Return the dataframe; it contains both original raw columns and the new model-ready columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (Binomial GLM) predicting mortgage approval (Accept)
    from applicant gender (Female) while controlling for creditworthiness and other covariates.

    The model includes an interaction Female x Black to test whether the gender effect
    differs for Black applicants.

    Returns a dict with the fitted statsmodels result object, odds ratios, and
    exponentiated confidence intervals (odds-ratio scale).
    """
    # Ensure the expected columns are present
    expected = [
        'Accept', 'Female', 'Black', 'Female_Black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z',
        'loan_to_value_z', 'housing_expense_ratio_z',
        'SelfEmployed', 'Married', 'BadHistory'
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Select predictors and add constant
    X_cols = [
        'Female', 'Black', 'Female_Black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z',
        'loan_to_value_z', 'housing_expense_ratio_z',
        'SelfEmployed', 'Married', 'BadHistory'
    ]

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['Accept']

    # Fit a Binomial (logistic) GLM
    model_fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Compute odds ratios and confidence intervals on the odds ratio scale
    odds_ratios = np.exp(model_fit.params)
    conf_int = model_fit.conf_int()
    conf_int.columns = ['2.5%', '97.5%']
    conf_odds = np.exp(conf_int)

    # Print summary for interactive use
    print(model_fit.summary())

    results = {
        'model': model_fit,
        'odds_ratios': odds_ratios,
        'conf_odds': conf_odds
    }

    return results


