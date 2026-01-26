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
    Transform the raw Boston HMDA lending dataset into a modeling-ready dataframe.

    Steps:
    - Make a copy to avoid modifying input
    - Ensure required columns exist
    - Drop rows missing the outcome or the key IV (female)
    - Impute missing numeric predictors with median, binary/categorical predictors with mode
    - Create clear, consistently-named columns used in modeling
    - Standardize continuous predictors (z-score) and produce *_z columns
    - Return a dataframe containing only the columns used in the statistical model
    """
    df = df.copy()

    # Required original columns (as supplied in dataset schema)
    required_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio'
    ]

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Drop rows with missing outcome or missing gender
    df = df.dropna(subset=['accept', 'female'])

    # Impute continuous predictors with median
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        if df[c].isnull().any():
            median = df[c].median()
            df[c] = df[c].fillna(median)

    # Impute binary/categorical predictors with mode
    bin_cols = ['black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in bin_cols:
        if df[c].isnull().any():
            mode = df[c].mode()
            if len(mode) == 0:
                fill = 0
            else:
                fill = mode[0]
            df[c] = df[c].fillna(fill)

    # Create clean column names for modeling
    df['Accept'] = df['accept'].astype(int)
    df['Female'] = df['female'].astype(int)
    df['Black'] = df['black'].astype(int)
    df['SelfEmployed'] = df['self_employed'].astype(int)
    df['Married'] = df['married'].astype(int)
    df['BadHistory'] = df['bad_history'].astype(int)
    df['DeniedPMI'] = df['denied_PMI'].astype(int)

    # Standardize continuous predictors (z-score). Use population denominator (ddof=0) for consistency.
    df['MortgageCredit_z'] = (df['mortgage_credit'] - df['mortgage_credit'].mean()) / (df['mortgage_credit'].std(ddof=0) if df['mortgage_credit'].std(ddof=0) != 0 else 1)
    df['ConsumerCredit_z'] = (df['consumer_credit'] - df['consumer_credit'].mean()) / (df['consumer_credit'].std(ddof=0) if df['consumer_credit'].std(ddof=0) != 0 else 1)
    df['PI_ratio_z'] = (df['PI_ratio'] - df['PI_ratio'].mean()) / (df['PI_ratio'].std(ddof=0) if df['PI_ratio'].std(ddof=0) != 0 else 1)
    df['LoanToValue_z'] = (df['loan_to_value'] - df['loan_to_value'].mean()) / (df['loan_to_value'].std(ddof=0) if df['loan_to_value'].std(ddof=0) != 0 else 1)
    df['HousingExpenseRatio_z'] = (df['housing_expense_ratio'] - df['housing_expense_ratio'].mean()) / (df['housing_expense_ratio'].std(ddof=0) if df['housing_expense_ratio'].std(ddof=0) != 0 else 1)

    # Final columns to be used in the model
    final_cols = [
        'Accept',
        'Female',
        'Black',
        'Married',
        'SelfEmployed',
        'BadHistory',
        'DeniedPMI',
        'MortgageCredit_z',
        'ConsumerCredit_z',
        'PI_ratio_z',
        'LoanToValue_z',
        'HousingExpenseRatio_z'
    ]

    # Return only the columns needed for modeling (copy to avoid chained-assignment issues downstream)
    return df[final_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression model to estimate the effect of gender on mortgage acceptance,
    controlling for applicant characteristics. Returns the fitted statsmodels results object.

    Model specification (logit):
      Accept ~ Female + Black + Married + SelfEmployed + BadHistory + DeniedPMI
               + MortgageCredit_z + ConsumerCredit_z + PI_ratio_z + LoanToValue_z + HousingExpenseRatio_z

    The function expects the dataframe to contain exactly the columns created by transform().
    """
    # Ensure required columns exist
    required = [
        'Accept', 'Female', 'Black', 'Married', 'SelfEmployed', 'BadHistory', 'DeniedPMI',
        'MortgageCredit_z', 'ConsumerCredit_z', 'PI_ratio_z', 'LoanToValue_z', 'HousingExpenseRatio_z'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define endogenous and exogenous variables
    y = df['Accept']
    X = df[[
        'Female', 'Black', 'Married', 'SelfEmployed', 'BadHistory', 'DeniedPMI',
        'MortgageCredit_z', 'ConsumerCredit_z', 'PI_ratio_z', 'LoanToValue_z', 'HousingExpenseRatio_z'
    ]]

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (use GLM with binomial family for robust behavior)
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        results = model.fit()
    except Exception:
        # fallback to Logit if GLM fails for any reason
        model = sm.Logit(y, X)
        results = model.fit(disp=False)

    # Print a concise summary and return the fitted results object
    print(results.summary())
    return results


