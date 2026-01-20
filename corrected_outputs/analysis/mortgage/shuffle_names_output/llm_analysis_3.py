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
    # Work on a copy
    df = df.copy()

    # ---------- DERIVE FEMALE (Independent variable) ----------
    # Priority 1: 'consumer_credit' sometimes encodes gender as 1=female,0=male in this dataset variant
    if 'consumer_credit' in df.columns:
        try:
            uniq = pd.Series(df['consumer_credit'].dropna().unique())
            # if values are binary 0/1, use it directly
            if set(uniq.tolist()).issubset({0, 1}):
                df['Female'] = df['consumer_credit'].astype(float).round().astype('Int64')
            else:
                # fallback: coerce to numeric and threshold at 0.5
                df['Female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').ge(0.5).astype('Int64')
        except Exception:
            df['Female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').ge(0.5).astype('Int64')
    elif 'female' in df.columns:
        # If there's a 'female' column, it may be 0/1 or continuous; threshold at 0.5
        df['Female'] = pd.to_numeric(df['female'], errors='coerce').ge(0.5).astype('Int64')
    else:
        # create column but it will be all NA; model will drop these rows
        df['Female'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')

    # ---------- DERIVE Approved (Dependent variable) ----------
    # Priority 1: 'Unnamed: 0' (some versions store approval as this column: 1 accepted, 0 denied)
    if 'Unnamed: 0' in df.columns:
        try:
            uniq = pd.Series(df['Unnamed: 0'].dropna().unique())
            if set(uniq.tolist()).issubset({0, 1}):
                df['Approved'] = df['Unnamed: 0'].astype(float).round().astype('Int64')
            else:
                # coerce: treat >0.5 as approved
                df['Approved'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce').ge(0.5).astype('Int64')
        except Exception:
            df['Approved'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce').ge(0.5).astype('Int64')
    # Priority 2: 'mortgage_credit' described in this schema as 1 if denied, 0 if accepted -> Approved = 1 - mortgage_credit
    elif 'mortgage_credit' in df.columns:
        m = pd.to_numeric(df['mortgage_credit'], errors='coerce')
        # If values are 0/1, assume 1=denied -> Approved = 1 - mortgage_credit
        if set(m.dropna().unique()).issubset({0, 1}):
            df['Approved'] = (1 - m).astype('Int64')
        else:
            # If not strictly 0/1, coerce and treat values <0.5 as approved (heuristic)
            df['Approved'] = m.lt(0.5).astype('Int64')
    else:
        # No clear approval column found; create column of NA
        df['Approved'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')

    # ---------- CONTROLS ----------
    # BadHistory: prefer 'bad_history' (schema suggests it encodes a bad credit flag or race in variants)
    if 'bad_history' in df.columns:
        df['BadHistory'] = pd.to_numeric(df['bad_history'], errors='coerce').round().astype('Int64')
    else:
        df['BadHistory'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')

    # Black: prefer an explicit 'black' / 'bad_history' if it encodes race
    if 'black' in df.columns:
        # If black appears continuous between 0 and 1, try to binarize at 0.5
        b = pd.to_numeric(df['black'], errors='coerce')
        if set(b.dropna().unique()).issubset({0, 1}):
            df['Black'] = b.astype('Int64')
        else:
            df['Black'] = b.ge(0.5).astype('Int64')
    else:
        # If 'bad_history' likely encodes race in this schema, copy it as Black (keeps consistency)
        if 'bad_history' in df.columns:
            df['Black'] = df['BadHistory']
        else:
            df['Black'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')

    # Married
    if 'married' in df.columns:
        df['Married'] = pd.to_numeric(df['married'], errors='coerce').round().astype('Int64')
    else:
        df['Married'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')

    # Self employed
    if 'self_employed' in df.columns:
        df['SelfEmployed'] = pd.to_numeric(df['self_employed'], errors='coerce').round().astype('Int64')
    else:
        df['SelfEmployed'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')

    # Loan to value
    if 'loan_to_value' in df.columns:
        df['LoanToValue'] = pd.to_numeric(df['loan_to_value'], errors='coerce')
    else:
        df['LoanToValue'] = pd.Series([pd.NA] * len(df), index=df.index)

    # PI ratio
    if 'PI_ratio' in df.columns:
        df['PI_ratio'] = pd.to_numeric(df['PI_ratio'], errors='coerce')
    else:
        df['PI_ratio'] = pd.Series([pd.NA] * len(df), index=df.index)

    # Housing expense ratio
    if 'housing_expense_ratio' in df.columns:
        df['HousingExpenseRatio'] = pd.to_numeric(df['housing_expense_ratio'], errors='coerce')
    else:
        df['HousingExpenseRatio'] = pd.Series([pd.NA] * len(df), index=df.index)

    # Denied PMI (continuous)
    if 'denied_PMI' in df.columns:
        df['Denied_PMI'] = pd.to_numeric(df['denied_PMI'], errors='coerce')
    else:
        df['Denied_PMI'] = pd.Series([pd.NA] * len(df), index=df.index)

    # ---------- FINAL HOUSEKEEPING ----------
    # Ensure columns exist with the exact names used in modeling
    final_cols = ['Approved', 'Female', 'BadHistory', 'Black', 'Married', 'SelfEmployed',
                  'LoanToValue', 'PI_ratio', 'HousingExpenseRatio', 'Denied_PMI']
    for c in final_cols:
        if c not in df.columns:
            df[c] = pd.Series([pd.NA] * len(df), index=df.index)

    # Return the transformed dataframe. The model function will select and drop NA rows as needed.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Build model dataframe with the exact columns defined above
    cols = ['Approved', 'Female', 'BadHistory', 'Black', 'Married', 'SelfEmployed',
            'LoanToValue', 'PI_ratio', 'HousingExpenseRatio', 'Denied_PMI']
    df_model = df[cols].copy()

    # Convert integer-like nullable columns to numeric (0/1) and keep floats for ratios
    for bcol in ['Approved', 'Female', 'BadHistory', 'Black', 'Married', 'SelfEmployed']:
        if bcol in df_model.columns:
            df_model[bcol] = pd.to_numeric(df_model[bcol], errors='coerce')

    for fcol in ['LoanToValue', 'PI_ratio', 'HousingExpenseRatio', 'Denied_PMI']:
        if fcol in df_model.columns:
            df_model[fcol] = pd.to_numeric(df_model[fcol], errors='coerce')

    # Drop rows with missing DV or IV (these are required). Controls missingness will also be dropped for simplicity.
    df_model = df_model.dropna(subset=['Approved', 'Female'])

    # If no variation in Approved or Female after dropping NA, return None
    if df_model['Approved'].nunique() < 2:
        raise ValueError('No variation in Approved variable after cleaning; cannot fit model.')
    if df_model['Female'].nunique() < 2:
        raise ValueError('No variation in Female variable after cleaning; cannot fit model.')

    # Choose control variables that have at least some non-missing values
    possible_controls = ['BadHistory', 'Black', 'Married', 'SelfEmployed',
                         'LoanToValue', 'PI_ratio', 'HousingExpenseRatio', 'Denied_PMI']
    # Keep controls that are not all-NA
    controls = [c for c in possible_controls if df_model[c].notna().sum() > 0]

    # Drop rows missing any of the selected controls (complete-case for this logistic model)
    model_cols = ['Approved', 'Female'] + controls
    df_model = df_model[model_cols].dropna()

    # Prepare design matrices
    y = df_model['Approved'].astype(float)
    X = df_model[['Female'] + controls].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (binary outcome)
    logit_model = sm.Logit(y, X)
    # Use try/except to catch convergence issues
    try:
        res = logit_model.fit(disp=False)
    except Exception:
        # Try with different method
        res = logit_model.fit(disp=False, method='bfgs', maxiter=100)

    # Attach some useful summaries: odds ratios and 95% CI
    try:
        params = res.params
        conf = res.conf_int()
        or_table = pd.DataFrame({
            'OR': np.exp(params),
            'CI_lower': np.exp(conf[0]),
            'CI_upper': np.exp(conf[1])
        })
    except Exception:
        or_table = None

    # Return a dict with the fitted result and odds ratios table for convenience
    return {
        'result': res,
        'odds_ratios': or_table,
        'model_columns': model_cols,
        'n_obs': int(res.nobs)
    }


