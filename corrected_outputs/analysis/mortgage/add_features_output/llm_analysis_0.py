from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/add_features_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw mortgage dataset into a cleaned dataframe containing the
    dependent variable, the main independent variable (female), and the control
    variables. Continuous controls are standardized (z-scored) to aid model
    convergence and interpretation of coefficients on comparable scales.

    Final columns required by the model (all created/preserved here):
      - deny
      - female
      - black
      - mortgage_credit_z
      - consumer_credit_z
      - PI_ratio_z
      - loan_to_value_z
      - bad_history
      - self_employed
      - married
      - housing_expense_ratio_z
    """

    df = df.copy()

    # Columns required for analysis
    required_cols = [
        'deny', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'PI_ratio', 'loan_to_value', 'bad_history', 'self_employed',
        'married', 'housing_expense_ratio'
    ]

    # Convert expected numeric/binary columns to numeric, coercing errors to NaN
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values on any variable required for the model
    df = df.dropna(subset=required_cols)

    # Ensure binary indicators are integer 0/1
    binary_cols = ['deny', 'female', 'black', 'bad_history', 'self_employed', 'married']
    for b in binary_cols:
        # Round to nearest integer then cast; this protects against float 0.0/1.0
        df[b] = df[b].round().astype(int)

    # Standardize continuous/ordinal controls (z-score). Use population sd (ddof=0)
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        zcol = c + '_z'
        if std == 0 or np.isnan(std):
            # If no variation, set standardized column to 0
            df[zcol] = 0.0
        else:
            df[zcol] = (df[c] - mean) / std

    # Keep only model-relevant columns (plus original columns if desired)
    final_cols = [
        'deny', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
        'PI_ratio_z', 'loan_to_value_z', 'bad_history', 'self_employed',
        'married', 'housing_expense_ratio_z'
    ]

    # Ensure all final columns exist
    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after transform: {missing}")

    # Return dataframe (keeping all original columns is fine, model will select needed ones)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (logit) predicting the probability that an
    application is denied (deny=1) from applicant gender (female) controlling
    for observed creditworthiness and demographic covariates.

    Returns a dictionary with the fitted Logit results and the average
    marginal effects summary for more interpretable effect sizes.
    """

    # Copy to avoid mutating input
    df = df.copy()

    # Define predictors used in the model (these must match transform output names)
    predictors = [
        'female', 'black', 'bad_history', 'self_employed', 'married',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z',
        'housing_expense_ratio_z'
    ]

    # Confirm predictors present
    missing = [p for p in predictors + ['deny'] if p not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')
    y = df['deny']

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X).fit(disp=False)

    # Compute average marginal effects (AME) for more intuitive effect sizes
    try:
        marg_eff = logit_model.get_margeff(at='overall')
    except Exception:
        # If margeff fails (rare), set to None but still return the logit result
        marg_eff = None

    results = {
        'logit_result': logit_model,
        'marginal_effects': marg_eff
    }

    return results


