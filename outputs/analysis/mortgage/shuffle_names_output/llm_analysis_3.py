from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # --- Gender (IV): create a clear binary 'is_female' ---
    if 'consumer_credit' in df.columns:
        # Schema documents consumer_credit as 1 if female, 0 if male
        df['is_female'] = pd.to_numeric(df['consumer_credit'], errors='coerce')
        # Force to exact 0/1
        df['is_female'] = df['is_female'].round().clip(lower=0, upper=1)
    elif 'female' in df.columns:
        # If only 'female' float-like column exists, threshold at 0.5
        df['is_female'] = pd.to_numeric(df['female'], errors='coerce')
        df['is_female'] = (df['is_female'] > 0.5).astype(float)
    else:
        df['is_female'] = np.nan

    # --- Dependent: approved (1 accepted, 0 denied) ---
    # Prefer 'Unnamed: 0' per schema description (1 accepted, 0 denied)
    if 'Unnamed: 0' in df.columns:
        df['approved'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce').round().clip(lower=0, upper=1)
    elif 'mortgage_credit' in df.columns:
        # schema says mortgage_credit == 1 indicates application was denied, 0 accepted
        mc = pd.to_numeric(df['mortgage_credit'], errors='coerce')
        df['approved'] = (1 - mc).round().clip(lower=0, upper=1)
    else:
        # try infer from other columns: if there is an 'accept' column with categories, treat >0 as accepted
        if 'accept' in df.columns:
            # Many datasets encode accept as 1..6; treat non-missing as accepted (1)
            df['approved'] = pd.to_numeric(df['accept'], errors='coerce').notnull().astype(float)
        else:
            df['approved'] = np.nan

    # --- Controls: create consistent columns (use available raw columns) ---
    # credit_score from 'accept' (schema mentions 'accept' as consumer credit score / rating)
    if 'accept' in df.columns:
        df['credit_score'] = pd.to_numeric(df['accept'], errors='coerce')
    else:
        df['credit_score'] = np.nan

    # loan_to_value (already 0..1 in schema)
    if 'loan_to_value' in df.columns:
        df['loan_to_value'] = pd.to_numeric(df['loan_to_value'], errors='coerce')
    else:
        df['loan_to_value'] = np.nan

    # debt-to-income or similar: use 'denied_PMI' as a numeric proxy per schema notes
    if 'denied_PMI' in df.columns:
        df['dti'] = pd.to_numeric(df['denied_PMI'], errors='coerce')
    elif 'PI_ratio' in df.columns:
        # PI_ratio may be a payment-to-income indicator in some variants
        df['dti'] = pd.to_numeric(df['PI_ratio'], errors='coerce')
    else:
        df['dti'] = np.nan

    # is_black from 'bad_history' (schema lists bad_history as 1 if applicant is Black)
    if 'bad_history' in df.columns:
        df['is_black'] = pd.to_numeric(df['bad_history'], errors='coerce').round().clip(lower=0, upper=1)
    elif 'black' in df.columns:
        # If 'black' is fractional or categorical, treat values > 0.5 as Black
        tmp = pd.to_numeric(df['black'], errors='coerce')
        df['is_black'] = (tmp > 0.5).astype(float)
    else:
        df['is_black'] = np.nan

    # is_married: use 'married' column if present
    if 'married' in df.columns:
        df['is_married'] = pd.to_numeric(df['married'], errors='coerce').round().clip(lower=0, upper=1)
    elif 'PI_ratio' in df.columns:
        # fallback: many schemas are inconsistent; leave as NaN if ambiguous
        df['is_married'] = np.nan
    else:
        df['is_married'] = np.nan

    # is_self_employed
    if 'self_employed' in df.columns:
        df['is_self_employed'] = pd.to_numeric(df['self_employed'], errors='coerce').round().clip(lower=0, upper=1)
    else:
        df['is_self_employed'] = np.nan

    # housing_expense_ratio (keep raw if present)
    if 'housing_expense_ratio' in df.columns:
        df['housing_expense_ratio'] = pd.to_numeric(df['housing_expense_ratio'], errors='coerce')
    else:
        df['housing_expense_ratio'] = np.nan

    # Keep only the standardized columns needed for modeling in the final DataFrame
    model_cols = [
        'approved', 'is_female', 'credit_score', 'loan_to_value', 'dti',
        'is_black', 'is_married', 'is_self_employed', 'housing_expense_ratio'
    ]

    # Ensure columns exist in the DataFrame (they should, from above); cast to numeric
    for c in model_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # Drop rows missing the DV or IV as they are essential for the analysis
    df = df.dropna(subset=['approved', 'is_female'])

    # For the regression we will drop rows with missing values in any model column
    df_model = df[model_cols].dropna()

    # Return the dataframe that contains all required modeling columns
    return df_model


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # df is expected to be the output of transform(): it contains the columns named below.
    # Build design matrix
    model_cols = [
        'is_female', 'credit_score', 'loan_to_value', 'dti',
        'is_black', 'is_married', 'is_self_employed', 'housing_expense_ratio'
    ]

    # Ensure required columns are present
    for c in ['approved'] + model_cols:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe passed to model().")

    # Dependent variable
    y = df['approved'].astype(float)

    # Independent variables (add constant)
    X = df[model_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (Logit)
    try:
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=False, method='lbfgs', maxiter=200)
    except Exception as e:
        # Fall back to GLM with binomial family if Logit fails to converge
        glm_mod = sm.GLM(y, X, family=sm.families.Binomial())
        result = glm_mod.fit()

    # Prepare a small summary output: coefficients and odds ratios
    coef = result.params
    conf = result.conf_int()
    conf.columns = ['ci_lower', 'ci_upper']
    odds_ratios = np.exp(coef)

    summary = {
        'model_result_object': result,
        'coefficients': coef.to_dict(),
        'conf_int': conf.to_dict(orient='index'),
        'odds_ratios': odds_ratios.to_dict(),
        'n_obs': int(result.nobs)
    }

    return summary


