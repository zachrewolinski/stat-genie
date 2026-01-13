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
    Transform raw HMDA-style dataset to the variables needed for logistic regression.

    Steps performed:
    - Rename relevant columns to clear names used in modeling.
    - Drop rows with missing values in any variable used in the model.
    - Ensure binary columns are 0/1 integers.
    - Create an interaction column Female_Black for testing moderation.
    - Standardize continuous predictors (z-score) used in the model to improve numerical stability.

    The returned dataframe will contain at least the columns listed in the conceptual variables:
    ['Female', 'Black', 'Female_Black', 'Accepted', 'housing_expense_ratio_z',
     'self_employed', 'married', 'mortgage_credit_z', 'consumer_credit_z',
     'bad_history', 'PI_ratio_z', 'loan_to_value_z', 'denied_PMI']
    """
    # Work on a copy
    df = df.copy()

    # Standardize column names from the provided schema to model column names
    rename_map = {
        'female': 'Female',
        'black': 'Black',
        'housing_expense_ratio': 'housing_expense_ratio',
        'self_employed': 'self_employed',
        'married': 'married',
        'mortgage_credit': 'mortgage_credit',
        'consumer_credit': 'consumer_credit',
        'bad_history': 'bad_history',
        'PI_ratio': 'PI_ratio',
        'loan_to_value': 'loan_to_value',
        'denied_PMI': 'denied_PMI',
        'accept': 'Accepted',
        'deny': 'deny'
    }
    df = df.rename(columns=rename_map)

    # Columns required for the model
    required_cols = [
        'Female', 'Black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
        'loan_to_value', 'denied_PMI', 'Accepted'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Ensure binary variables are integer (0/1)
    for col in ['Female', 'Black', 'self_employed', 'married', 'bad_history', 'denied_PMI', 'Accepted']:
        # coerce to numeric then to int (assumes values are 0/1 or convertible)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=[col])
        df[col] = df[col].astype(int)

    # Create interaction term for moderation test (Female x Black)
    df['Female_Black'] = df['Female'] * df['Black']

    # Helper to compute z-score robustly (avoid division by zero)
    def zscore(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors='coerce')
        mean = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return s - mean  # will be zeros if constant
        return (s - mean) / std

    # Standardize continuous predictors and create new columns with _z suffix
    df['housing_expense_ratio_z'] = zscore(df['housing_expense_ratio'])
    df['mortgage_credit_z'] = zscore(df['mortgage_credit'])
    df['consumer_credit_z'] = zscore(df['consumer_credit'])
    df['PI_ratio_z'] = zscore(df['PI_ratio'])
    df['loan_to_value_z'] = zscore(df['loan_to_value'])

    # Keep only the columns needed for modeling (but leave the dataframe intact otherwise)
    model_cols = [
        'Accepted', 'Female', 'Black', 'Female_Black', 'housing_expense_ratio_z',
        'self_employed', 'married', 'mortgage_credit_z', 'consumer_credit_z',
        'bad_history', 'PI_ratio_z', 'loan_to_value_z', 'denied_PMI'
    ]

    # Ensure model columns exist (if not, error will surface downstream)
    missing = [c for c in model_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns after transform: {missing}")

    # Return dataframe with model columns plus any original columns (keeping only model columns is OK too)
    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (logit) model predicting loan acceptance from applicant gender
    controlling for applicant characteristics. Also includes a Female x Black interaction to test
    whether the gender effect differs for Black applicants.

    Returns a dictionary containing the fitted statsmodels result object, odds ratios, and
    95% confidence intervals for the odds ratios.
    """
    # Prepare dependent and independent variables
    y = df['Accepted']

    X_cols = [
        'Female',        # primary IV
        'Black',
        'Female_Black',  # interaction (moderation)
        # controls (binary left as-is)
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI',
        # standardized continuous controls
        'housing_expense_ratio_z',
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z'
    ]

    X = df[X_cols]

    # Add constant for intercept
    X_const = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X_const)
    # disable optimizer output by disp=False
    results = logit_model.fit(disp=False)

    # Compute odds ratios and 95% CI for them
    params = results.params
    conf = results.conf_int()
    conf.columns = ['2.5%', '97.5%']

    odds_ratios = np.exp(params)
    conf_int_exp = np.exp(conf)

    # Package results
    out = {
        'results': results,                 # statsmodels results object (has .summary())
        'odds_ratios': odds_ratios,         # pandas Series
        'odds_ratio_CI_95': conf_int_exp    # DataFrame with columns ['2.5%','97.5%']
    }

    return out


