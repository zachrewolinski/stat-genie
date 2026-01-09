from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/mortgage/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston Fed mortgage dataset into a dataframe suitable for logistic regression.

    - Coerce relevant columns to numeric
    - Drop rows with missing data in variables needed for the analysis
    - Create standardized (z-score) versions of continuous controls
    - Ensure binary variables are integer 0/1

    Returns the dataframe containing at least the columns used in the model:
      ['accept', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
       'PI_ratio_z', 'loan_to_value_z', 'bad_history', 'married', 'self_employed',
       'housing_expense_ratio_z']
    """
    # Work on a copy
    df = df.copy()

    # Columns required for the analysis (original names in raw df)
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'PI_ratio', 'loan_to_value', 'bad_history', 'married', 'self_employed',
        'housing_expense_ratio'
    ]

    # Coerce to numeric (in case of string floats) and handle missing values
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If a required column is missing entirely, raise a clear error
            raise KeyError(f"Required column '{col}' not found in dataframe")

    # Drop rows with any missing values in required columns
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are integers (0/1)
    bin_cols = ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed']
    for b in bin_cols:
        df[b] = df[b].astype(int)

    # Standardize continuous predictors (z-scores). Use sample std (ddof=0) for population-like standardization
    cont_map = {
        'mortgage_credit': 'mortgage_credit_z',
        'consumer_credit': 'consumer_credit_z',
        'PI_ratio': 'PI_ratio_z',
        'loan_to_value': 'loan_to_value_z',
        'housing_expense_ratio': 'housing_expense_ratio_z'
    }

    for orig, zname in cont_map.items():
        # If the column has zero variance, create zeros to avoid division by zero
        std = df[orig].std(ddof=0)
        mean = df[orig].mean()
        if std == 0 or np.isclose(std, 0.0):
            df[zname] = 0.0
        else:
            df[zname] = (df[orig] - mean) / std

    # Keep only the columns needed for modeling to make downstream code explicit
    model_cols = ['accept', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
                  'PI_ratio_z', 'loan_to_value_z', 'bad_history', 'married', 'self_employed',
                  'housing_expense_ratio_z']

    # Return dataframe with at least those columns (plus any others unchanged)
    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to estimate the effect of applicant gender on approval,
    controlling for creditworthiness and other applicant characteristics.

    Model specification:
      logit( P(accept=1) ) = beta0 + beta1 * female + beta2 * black + beta3 * mortgage_credit_z
                              + beta4 * consumer_credit_z + beta5 * PI_ratio_z + beta6 * loan_to_value_z
                              + beta7 * bad_history + beta8 * married + beta9 * self_employed
                              + beta10 * housing_expense_ratio_z

    Returns the fitted statsmodels results object.
    """
    # Ensure required columns exist in df
    required = ['accept', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
                'PI_ratio_z', 'loan_to_value_z', 'bad_history', 'married', 'self_employed',
                'housing_expense_ratio_z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe is missing required columns for modeling: {missing}")

    # Design matrix X and outcome y
    y = df['accept'].astype(int)
    X = df[[
        'female', 'black', 'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z',
        'loan_to_value_z', 'bad_history', 'married', 'self_employed', 'housing_expense_ratio_z'
    ]].astype(float)

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (use GLM with binomial family for stable output / robust compat)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    results = model.fit()

    # Print a brief summary to console (useful when running interactively); return results object
    print(results.summary())
    return results


