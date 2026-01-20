from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataset for modeling the effect of gender on mortgage approval.

    Transformations performed:
    - Coerce relevant columns to numeric
    - Construct Approved binary outcome (1 = accepted, 0 = denied) from 'mortgage_credit' where available
    - Construct Female binary indicator from 'consumer_credit' (>=0.5 treated as female)
    - Keep the control columns used in the model and drop rows with missing values in any model column

    Final dataframe contains the exact column names used in the model:
    ['Approved', 'Female', 'PI_ratio', 'loan_to_value', 'denied_PMI', 'self_employed', 'married', 'bad_history', 'housing_expense_ratio']
    """
    df = df.copy()

    # List of raw columns we will use or derive from
    raw_cols = [
        'consumer_credit',      # gender indicator: 1 female, 0 male
        'mortgage_credit',      # 1 = denied, 0 = accepted (per schema)
        'PI_ratio',
        'loan_to_value',
        'denied_PMI',
        'self_employed',
        'married',
        'bad_history',
        'housing_expense_ratio',
        'Unnamed: 0'
    ]

    # Coerce to numeric where present
    for c in raw_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Construct Approved outcome
    if 'mortgage_credit' in df.columns:
        # Schema indicates mortgage_credit: 1 if denied, 0 if accepted
        df['Approved'] = (df['mortgage_credit'] == 0).astype(int)
    elif 'Unnamed: 0' in df.columns:
        # Fallback: some versions encode acceptance in 'Unnamed: 0' (1 accepted, 0 denied)
        df['Approved'] = (df['Unnamed: 0'] == 1).astype(int)
    else:
        # If neither column exists, create Approved as NA so dropping will remove these rows
        df['Approved'] = pd.NA

    # Construct Female indicator from consumer_credit (schema: 1 female, 0 male)
    if 'consumer_credit' in df.columns:
        # treat any value >= 0.5 as female to accommodate float encodings
        df['Female'] = (df['consumer_credit'] >= 0.5).astype(int)
    else:
        # try alternative 'female' column only if it appears to be binary 0/1
        if 'female' in df.columns:
            df['Female'] = (pd.to_numeric(df['female'], errors='coerce') >= 0.5).astype(int)
        else:
            df['Female'] = pd.NA

    # Ensure control columns exist; if not present create NA columns so dropna will remove incomplete rows
    controls = ['PI_ratio', 'loan_to_value', 'denied_PMI', 'self_employed', 'married', 'bad_history', 'housing_expense_ratio']
    for c in controls:
        if c not in df.columns:
            df[c] = pd.NA

    # Keep only rows with non-missing values in all model columns
    model_cols = ['Approved', 'Female'] + controls
    df = df.dropna(subset=model_cols)

    # Cast final columns to numeric ints/floats explicitly
    df['Approved'] = df['Approved'].astype(int)
    df['Female'] = df['Female'].astype(int)
    for c in controls:
        # leave numeric controls as floats
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Final check: drop rows again if controls could not be coerced
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (maximum likelihood Logit) predicting probability of approval
    from applicant gender and control variables that capture creditworthiness and application
    characteristics.

    Model specification (primary):
      Approved ~ Female + PI_ratio + loan_to_value + denied_PMI + self_employed + married + bad_history + housing_expense_ratio

    Returns:
      The fitted statsmodels Logit results object.
    """
    # Columns used in the model (must match transform output)
    controls = ['PI_ratio', 'loan_to_value', 'denied_PMI', 'self_employed', 'married', 'bad_history', 'housing_expense_ratio']

    # Ensure required columns are present
    required = ['Approved', 'Female'] + controls
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Design matrix and outcome
    y = df['Approved'].astype(int)
    X = df[['Female'] + controls].astype(float)

    # Add constant (intercept)
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression using statsmodels
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Return the fitted results object so the caller can inspect .summary(), .params, .bse, etc.
    return results


