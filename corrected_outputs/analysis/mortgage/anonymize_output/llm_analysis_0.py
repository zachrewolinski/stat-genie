from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/anonymize_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with clearly named columns used for modeling.
    Inputs:
      - df: original dataframe containing columns feature1..feature14
    Returns:
      - df: transformed dataframe containing at least the columns referenced in the conceptual model
    """
    df = df.copy()

    # Rename relevant raw features to meaningful column names used in the model
    rename_map = {
        'feature2': 'female',                 # 1 if female, 0 if male
        'feature3': 'black',                  # 1 if Black, 0 otherwise
        'feature4': 'housing_expense_ratio',  # housing expense / income
        'feature5': 'self_employed',          # 1 if self-employed
        'feature6': 'married',                # 1 if married
        'feature7': 'mortgage_credit_score',  # mortgage credit score (ordinal)
        'feature8': 'consumer_credit_score',  # consumer credit score (ordinal)
        'feature9': 'bad_credit',             # 1 if history of bad credit
        'feature10': 'debt_to_income',        # total debt payments / income
        'feature11': 'denied',                # 1 if denied, 0 if accepted (redundant with feature14)
        'feature12': 'loan_to_value',         # loan amount / appraised value
        'feature13': 'pmi_denied',            # 1 if PMI denied, 0 otherwise
        'feature14': 'approved'               # 1 if accepted, 0 if denied
    }

    df = df.rename(columns=rename_map)

    # Keep only rows with a non-missing approval outcome and gender
    df = df.dropna(subset=['approved', 'female'])

    # Ensure binary indicators are numeric 0/1
    bin_cols = ['female', 'black', 'self_employed', 'married', 'bad_credit', 'pmi_denied', 'approved', 'denied']
    for c in bin_cols:
        if c in df.columns:
            # Some columns might be floats (0.0/1.0); coerce to integers safely
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # After coercion, drop rows with NAs in crucial binary columns
    df = df.dropna(subset=['female', 'approved'])
    df['female'] = df['female'].astype(int)
    df['approved'] = df['approved'].astype(int)

    # For the other binary controls, if missing drop those rows (they are important controls)
    other_binary_controls = ['black', 'self_employed', 'married', 'bad_credit', 'pmi_denied']
    present_bin_controls = [c for c in other_binary_controls if c in df.columns]
    if present_bin_controls:
        df = df.dropna(subset=present_bin_controls)
        for c in present_bin_controls:
            df[c] = df[c].astype(int)

    # Continuous/ordinal predictors: ensure numeric and drop rows with missing values for key numeric controls
    numeric_controls = ['mortgage_credit_score', 'consumer_credit_score', 'debt_to_income', 'loan_to_value', 'housing_expense_ratio']
    present_numeric_controls = [c for c in numeric_controls if c in df.columns]
    if present_numeric_controls:
        for c in present_numeric_controls:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=present_numeric_controls)

    # Standardize continuous predictors (create _z columns). Keep originals too.
    for c in present_numeric_controls:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # avoid division by zero
        if std == 0 or pd.isna(std):
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Final sanity: ensure final columns exist
    final_cols = [
        'female', 'approved', 'black',
        'mortgage_credit_score_z', 'consumer_credit_score_z',
        'debt_to_income_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'bad_credit', 'self_employed', 'married', 'pmi_denied'
    ]
    # Keep only columns that actually exist in df to avoid KeyErrors downstream; but we have ensured presence above
    final_present = [c for c in final_cols if c in df.columns]

    # Return the full dataframe (including new standardized columns). Modeling function will subset and drop NA as needed.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting mortgage approval from applicant gender
    while adjusting for relevant credit and demographic controls.

    Returns the fitted statsmodels results object (Logit or GLM fallback).
    """
    import statsmodels.api as sm

    # Columns to use as predictors (must match columns created in transform)
    predictor_cols = [
        'female',
        'black',
        'mortgage_credit_score_z',
        'consumer_credit_score_z',
        'debt_to_income_z',
        'loan_to_value_z',
        'housing_expense_ratio_z',
        'bad_credit',
        'self_employed',
        'married',
        'pmi_denied'
    ]

    # Keep only the columns present in df (transform should have created them). Drop rows with missing values in those cols.
    present_predictors = [c for c in predictor_cols if c in df.columns]
    model_df = df.dropna(subset=['approved'] + present_predictors).copy()

    if model_df.shape[0] == 0:
        raise ValueError('No rows available for modeling after dropping missing values. Check transform output.')

    X = model_df[present_predictors]
    X = sm.add_constant(X, has_constant='add')
    y = model_df['approved']

    # Fit logistic regression. Use Logit but fall back to GLM(Binomial) if convergence or separation issues occur.
    try:
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=False, method='lbfgs', maxiter=200)
    except Exception as e:
        # fallback to GLM with binomial (more robust to some issues)
        glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
        result = glm_binom.fit()

    # Return the fitted model result object. Users can call result.summary(), result.params, etc.
    return result


