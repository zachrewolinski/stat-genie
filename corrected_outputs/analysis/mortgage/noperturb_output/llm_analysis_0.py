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
    """
    Prepare data for logistic regression of mortgage acceptance on gender.

    Steps:
    - Keep and coerce relevant columns to numeric
    - Drop rows missing the DV, IV, or any controls used in the model
    - Ensure binary columns are integers
    - Create female x black interaction to test moderation
    - Standardize continuous covariates (z-score) for stable estimation and interpretation
    - Return a dataframe containing only the final variables used in the model
    """
    df = df.copy()

    # Columns we plan to use
    needed = [
        'female',
        'accept',
        'black',
        'housing_expense_ratio',
        'self_employed',
        'married',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'denied_PMI'
    ]

    # Coerce to numeric where present
    for c in needed:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the dependent variable or main independent variable
    df = df.dropna(subset=['accept', 'female'])

    # Drop rows missing any of the controls we plan to include (complete-case analysis)
    present_controls = [c for c in needed if c in df.columns]
    df = df.dropna(subset=present_controls)

    # Ensure binary variables are integers
    df['female'] = df['female'].astype(int)
    df['accept'] = df['accept'].astype(int)
    if 'black' in df.columns:
        df['black'] = df['black'].astype(int)

    for b in ['self_employed', 'married', 'bad_history', 'denied_PMI']:
        if b in df.columns:
            df[b] = df[b].astype(int)

    # Interaction term for moderation test (female x black)
    if 'black' in df.columns:
        df['female_black_interaction'] = df['female'] * df['black']

    # Standardize continuous covariates (z-scores). Create new columns with _z suffix.
    cont_vars = ['housing_expense_ratio', 'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value']
    for v in cont_vars:
        if v in df.columns:
            mean = df[v].mean()
            std = df[v].std()
            if std == 0 or np.isnan(std):
                # if no variance, set to zero (constant)
                df[v + '_z'] = 0.0
            else:
                df[v + '_z'] = (df[v] - mean) / std

    # Final set of columns to return (only those that exist in the input)
    final_cols = [
        'female',
        'accept',
        'black',
        'female_black_interaction',
        'housing_expense_ratio_z',
        'self_employed',
        'married',
        'mortgage_credit_z',
        'consumer_credit_z',
        'bad_history',
        'PI_ratio_z',
        'loan_to_value_z',
        'denied_PMI'
    ]

    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting acceptance (accept) from female (gender),
    controls, and a female x black interaction to test moderation by race.

    Returns the fitted statsmodels results object (Logit).
    """
    # Copy to avoid modifying caller's dataframe
    df = df.copy()

    # Define predictors in the order used for interpretation
    predictors = []
    # core IV
    predictors.append('female')
    # moderator and interaction if present
    if 'black' in df.columns:
        predictors.append('black')
    if 'female_black_interaction' in df.columns:
        predictors.append('female_black_interaction')
    # continuous standardized controls
    for c in ['housing_expense_ratio_z', 'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z']:
        if c in df.columns:
            predictors.append(c)
    # binary controls
    for c in ['self_employed', 'married', 'bad_history', 'denied_PMI']:
        if c in df.columns:
            predictors.append(c)

    # Prepare design matrix
    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (maximum likelihood)
    # Use GLM with binomial family or Logit; Logit is used here.
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Return the fitted results object so caller can inspect params, summary, conf_int, etc.
    return results


