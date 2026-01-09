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
    Transform the raw dataset into a dataframe containing the variables used in the model:
      - approved: binary outcome 1 if application accepted, 0 if denied
      - is_female: binary indicator 1 for female, 0 for male
      - credit_score: numeric/ordinal credit score from 'accept'
      - PI_ratio, loan_to_value, denied_PMI, self_employed, married, bad_history as controls

    The code coercively converts relevant columns to numeric, derives variables, and drops rows with missing values in the final set of variables.
    """
    df = df.copy()

    # Coerce likely-relevant columns to numeric (if present)
    cols_to_numeric = [
        'consumer_credit',  # documented as 1 if female, 0 if male
        'mortgage_credit',  # documented as 1 if denied, 0 if accepted
        'Unnamed: 0',       # alternate accept/deny indicator in some versions
        'accept',           # documented as consumer credit score / applicant credit score
        'PI_ratio',
        'loan_to_value',
        'denied_PMI',
        'self_employed',
        'married',
        'bad_history',
        'female'
    ]
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Dependent variable: approved (1 accepted, 0 denied)
    # Prefer 'mortgage_credit' where documented as 1=denied,0=accepted
    if 'mortgage_credit' in df.columns:
        # mortgage_credit: 1 if denied, 0 if accepted
        df['approved'] = (df['mortgage_credit'] == 0).astype('int')
    elif 'Unnamed: 0' in df.columns:
        # fallback: some descriptions indicate Unnamed: 0 is 1 accepted, 0 denied
        df['approved'] = df['Unnamed: 0'].astype('Int64').fillna(-1).astype('int')
        # assume already 1=accepted,0=denied; keep as-is
    else:
        raise KeyError("Neither 'mortgage_credit' nor 'Unnamed: 0' found in dataframe to derive approval outcome.")

    # Independent variable: is_female
    if 'consumer_credit' in df.columns:
        # documented as 1 if female, 0 if male
        df['is_female'] = df['consumer_credit'].astype('Int64').fillna(-1).astype('int')
    elif 'female' in df.columns:
        # fallback: threshold the 'female' column at 0.5 (makes 1 ~ female, 0 ~ male) if the column is continuous
        df['is_female'] = (df['female'] > 0.5).astype('int')
    else:
        raise KeyError("Neither 'consumer_credit' nor 'female' present to derive gender indicator.")

    # Controls: credit_score (from 'accept') and other financial / demographic controls
    if 'accept' in df.columns:
        df['credit_score'] = pd.to_numeric(df['accept'], errors='coerce')
    else:
        # if absent, create a missing column (will drop later if necessary)
        df['credit_score'] = pd.NA

    # Ensure control columns exist (if missing, create as NA so dropna can handle consistently)
    for col in ['PI_ratio', 'loan_to_value', 'denied_PMI', 'self_employed', 'married', 'bad_history']:
        if col not in df.columns:
            df[col] = pd.NA

    # Select final columns needed for modeling
    final_cols = [
        'approved',
        'is_female',
        'credit_score',
        'PI_ratio',
        'loan_to_value',
        'denied_PMI',
        'self_employed',
        'married',
        'bad_history'
    ]

    # Drop rows with missing values in any of the model variables (outcome + predictors)
    df = df.dropna(subset=final_cols)

    # Coerce types: integer/binary columns to int
    for bcol in ['approved', 'is_female', 'self_employed', 'married', 'bad_history']:
        if bcol in df.columns:
            # convert to integer 0/1 (if values aren't exactly 0/1, they remain numeric but coerced)
            df[bcol] = df[bcol].astype(int)

    # Final dataframe contains at least the columns in final_cols
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting approval (approved) from applicant gender (is_female)
    controlling for creditworthiness and other covariates.

    Model: logit( P(approved=1) ) = beta0 + beta1*is_female + beta2*credit_score + beta3*PI_ratio
                                    + beta4*loan_to_value + beta5*denied_PMI + beta6*self_employed
                                    + beta7*married + beta8*bad_history

    Returns: fitted statsmodels binary model results object.
    """
    # Ensure required columns exist
    required = ['approved', 'is_female', 'credit_score', 'PI_ratio', 'loan_to_value', 'denied_PMI', 'self_employed', 'married', 'bad_history']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Prepare X and y
    y = df['approved'].astype(int)
    X = df[['is_female', 'credit_score', 'PI_ratio', 'loan_to_value', 'denied_PMI', 'self_employed', 'married', 'bad_history']].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood estimation)
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Return the fitted results object (user can inspect .summary() or .params etc.)
    return results


