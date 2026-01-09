from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/mortgage/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for modeling gender effects on mortgage denial.

    Produces the following columns (exact names used by the model):
      - female: 0/1 indicator for female applicant
      - mortgage_denied: 0/1 indicator where 1 = denied, 0 = accepted
      - credit_score: numeric/ordinal credit score from 'accept'
      - loan_to_value, housing_expense_ratio, PI_ratio, self_employed, married, black, denied_PMI

    The function is defensive: it will try to map multiple possible source columns to the required final columns when the schema is ambiguous.
    """
    df = df.copy()

    # --- Create female indicator ---
    if 'consumer_credit' in df.columns:
        # Schema: consumer_credit described as 1 if applicant is female, 0 if male
        df['female'] = pd.to_numeric(df['consumer_credit'], errors='coerce')
    elif 'female' in df.columns:
        # Some datasets might already have a female column; coerce to numeric and map to 0/1
        df['female'] = pd.to_numeric(df['female'], errors='coerce')
    else:
        # If no gender column is present, create NA column so downstream will drop
        df['female'] = np.nan

    # Normalize female to strict 0/1 when possible
    df['female'] = df['female'].where(df['female'].isna(), df['female'].astype(float))
    df.loc[~df['female'].isna(), 'female'] = df.loc[~df['female'].isna(), 'female'].apply(lambda x: 1 if float(x) >= 0.5 else 0)

    # --- Create mortgage_denied dependent variable ---
    # Prefer a clearly described column 'mortgage_credit' (schema: 1 = denied, 0 = accepted)
    if 'mortgage_credit' in df.columns:
        df['mortgage_denied'] = pd.to_numeric(df['mortgage_credit'], errors='coerce')
    elif 'deny' in df.columns:
        # 'deny' sometimes holds counts or indicators; attempt to coerce to binary (non-zero -> denied)
        tmp = pd.to_numeric(df['deny'], errors='coerce')
        df['mortgage_denied'] = tmp.apply(lambda x: 1 if (not pd.isna(x) and x != 0) else (0 if not pd.isna(x) else np.nan))
    elif 'Unnamed: 0' in df.columns:
        # In some provided descriptions 'Unnamed: 0' was used as acceptance flag; try to interpret
        tmp = pd.to_numeric(df['Unnamed: 0'], errors='coerce')
        # If Unnamed: 0 encodes 1 = accepted, 0 = denied (ambiguous), we try to detect and convert.
        # We assume 1 -> accepted, 0 -> denied -> so mortgage_denied = 1 - Unnamed: 0
        df['mortgage_denied'] = tmp.apply(lambda x: (1 - x) if not pd.isna(x) else np.nan)
    else:
        df['mortgage_denied'] = np.nan

    # Ensure mortgage_denied is numeric 0/1 when possible
    df['mortgage_denied'] = df['mortgage_denied'].where(df['mortgage_denied'].isna(), pd.to_numeric(df['mortgage_denied'], errors='coerce').astype(float))
    df.loc[~df['mortgage_denied'].isna(), 'mortgage_denied'] = df.loc[~df['mortgage_denied'].isna(), 'mortgage_denied'].apply(lambda x: 1 if float(x) >= 0.5 else 0)

    # --- Controls: coerce and rename from schema columns ---
    # credit_score from 'accept'
    if 'accept' in df.columns:
        df['credit_score'] = pd.to_numeric(df['accept'], errors='coerce')
    else:
        df['credit_score'] = np.nan

    # loan_to_value
    if 'loan_to_value' in df.columns:
        df['loan_to_value'] = pd.to_numeric(df['loan_to_value'], errors='coerce')
    else:
        df['loan_to_value'] = np.nan

    # housing_expense_ratio
    if 'housing_expense_ratio' in df.columns:
        df['housing_expense_ratio'] = pd.to_numeric(df['housing_expense_ratio'], errors='coerce')
    else:
        df['housing_expense_ratio'] = np.nan

    # PI_ratio (payment-to-income or similar)
    if 'PI_ratio' in df.columns:
        df['PI_ratio'] = pd.to_numeric(df['PI_ratio'], errors='coerce')
    else:
        df['PI_ratio'] = np.nan

    # self_employed
    if 'self_employed' in df.columns:
        df['self_employed'] = pd.to_numeric(df['self_employed'], errors='coerce')
    else:
        df['self_employed'] = np.nan

    # married
    if 'married' in df.columns:
        df['married'] = pd.to_numeric(df['married'], errors='coerce')
    else:
        df['married'] = np.nan

    # black (race) -- schema indicates 'bad_history' was the Black indicator in this dataset description
    if 'bad_history' in df.columns:
        df['black'] = pd.to_numeric(df['bad_history'], errors='coerce')
    elif 'black' in df.columns:
        df['black'] = pd.to_numeric(df['black'], errors='coerce')
    else:
        df['black'] = np.nan

    # denied_PMI
    if 'denied_PMI' in df.columns:
        df['denied_PMI'] = pd.to_numeric(df['denied_PMI'], errors='coerce')
    else:
        df['denied_PMI'] = np.nan

    # --- Final housekeeping: drop rows missing the DV or the main IV ---
    required_for_model = ['mortgage_denied', 'female', 'credit_score', 'loan_to_value', 'housing_expense_ratio', 'PI_ratio', 'self_employed', 'married', 'black', 'denied_PMI']

    # It's often better to keep as many rows as possible; however, the model requires no NA in predictors and outcome.
    df = df.dropna(subset=required_for_model)

    # Convert columns to appropriate dtypes (int for binary indicators)
    df['female'] = df['female'].astype(int)
    df['mortgage_denied'] = df['mortgage_denied'].astype(int)
    df['married'] = df['married'].apply(lambda x: int(x) if (not pd.isna(x) and float(x) in [0,1]) else int(x) if not pd.isna(x) else x)
    df['self_employed'] = df['self_employed'].apply(lambda x: int(x) if (not pd.isna(x) and float(x) in [0,1]) else int(x) if not pd.isna(x) else x)

    # Ensure numeric types for continuous predictors
    numeric_cols = ['credit_score', 'loan_to_value', 'housing_expense_ratio', 'PI_ratio', 'denied_PMI', 'black']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Return the dataframe containing all columns necessary for the model (plus any original columns still present).
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting mortgage denial from applicant gender controlling for credit and application covariates.

    Returns a dictionary with keys:
      - 'fit': the fitted statsmodels results object
      - 'marginal_effects': average marginal effects dataframe (if computable)

    Exact predictor column names used (must exist in df returned by transform):
      female, credit_score, loan_to_value, housing_expense_ratio, PI_ratio, self_employed, married, black, denied_PMI
    Dependent variable: mortgage_denied
    """
    df = df.copy()

    # Columns to use
    predictors = ['female', 'credit_score', 'loan_to_value', 'housing_expense_ratio', 'PI_ratio', 'self_employed', 'married', 'black', 'denied_PMI']
    outcome = 'mortgage_denied'

    # Ensure no missing data for the model
    model_df = df.dropna(subset=predictors + [outcome]).copy()

    # Design matrix
    X = model_df[predictors]
    X = sm.add_constant(X, has_constant='add')
    y = model_df[outcome]

    # Fit logistic regression (Logit). If Logit fails to converge/use, fall back to GLM with binomial family.
    try:
        logit_res = sm.Logit(y, X).fit(disp=False)
    except Exception:
        logit_res = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Robust (heteroskedasticity-consistent) covariance if available
    try:
        # Recompute fit with robust covariance for more reliable SEs
        logit_res_robust = logit_res.get_robustcov_results(cov_type='HC3')
    except Exception:
        logit_res_robust = logit_res

    # Average marginal effects for the female indicator and other predictors (if supported)
    try:
        margeff = logit_res_robust.get_margeff(at='overall', method='dydx')
        margeff_df = margeff.summary_frame()
    except Exception:
        margeff_df = None

    results = {
        'fit': logit_res_robust,
        'marginal_effects': margeff_df,
        'predictors': predictors,
        'outcome': outcome,
        'n_obs': int(model_df.shape[0])
    }

    return results

