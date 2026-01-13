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
    """
    Transform the raw dataframe into the modeling dataframe. The function creates the following columns used in the model:
      - Approved (DV): 1 if mortgage application accepted, 0 if denied
      - Female (IV): 1 if applicant is female, 0 if male
      - Black (control): 1 if applicant is Black, 0 otherwise
      - Standardized continuous controls: PI_ratio_z, loan_to_value_z, housing_expense_ratio_z, denied_PMI_z
      - Binary controls kept as integers: self_employed, married

    The function tries multiple columns as fallbacks because the input schema contains multiple overlapping/ambiguous columns.
    """

    df = df.copy()

    # --- Dependent variable: Approved ---
    if 'Unnamed: 0' in df.columns:
        # dataset schema example: Unnamed: 0 described as 1 if accepted, 0 if denied
        df['Approved'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce').astype('float')
    elif 'mortgage_credit' in df.columns:
        # mortgage_credit described as 1 if denied, 0 if accepted -> Approved = 1 - mortgage_credit
        df['Approved'] = 1 - pd.to_numeric(df['mortgage_credit'], errors='coerce').astype('float')
    elif 'accept' in df.columns:
        # 'accept' may be a multi-level field in some schemas (1..6). If it's binary use it; otherwise try to infer acceptance.
        uniq = pd.Series(df['accept']).dropna().unique()
        if set(uniq).issubset({0, 1}):
            df['Approved'] = pd.to_numeric(df['accept'], errors='coerce').astype('float')
        else:
            # fallback: if higher values indicate acceptance we treat values >= median as Approved
            df['Approved'] = (pd.to_numeric(df['accept'], errors='coerce') >= pd.to_numeric(df['accept'], errors='coerce').median()).astype(float)
    else:
        # final fallback: if 'deny' exists and is binary-ish, use its inverse
        if 'deny' in df.columns and set(pd.Series(df['deny']).dropna().unique()).issubset({0, 1}):
            df['Approved'] = 1 - pd.to_numeric(df['deny'], errors='coerce').astype('float')
        else:
            raise KeyError("Could not find a suitable column to derive Approved. Expected one of ['Unnamed: 0','mortgage_credit','accept','deny'].")

    # Ensure Approved is binary 0/1 (allow small numeric noise caused by conversions)
    df['Approved'] = df['Approved'].map(lambda x: 1.0 if pd.notnull(x) and float(x) >= 0.5 else (0.0 if pd.notnull(x) else np.nan))

    # --- Independent variable: Female ---
    if 'consumer_credit' in df.columns:
        # schema note: consumer_credit described as 1 if applicant is female, 0 if male
        uniq = pd.Series(df['consumer_credit']).dropna().unique()
        if set(uniq).issubset({0, 1}):
            df['Female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').astype('float')
        else:
            # fallback if it's not strictly binary
            df['Female'] = (pd.to_numeric(df['consumer_credit'], errors='coerce') >= 0.5).astype(float)
    elif 'female' in df.columns:
        # try to interpret 'female' column: if values are near 0/1 use threshold, else try to rescale
        series = pd.to_numeric(df['female'], errors='coerce')
        # if most values are already 0/1
        uniq = series.dropna().unique()
        if set(uniq).issubset({0, 1}):
            df['Female'] = series.astype('float')
        else:
            # threshold at 0.5
            df['Female'] = (series > 0.5).astype(float)
    else:
        raise KeyError("Could not find a suitable column to derive Female. Expected 'consumer_credit' or 'female'.")

    # --- Control variables ---
    # Black (race) - prefer 'bad_history' if present, otherwise try 'black'
    if 'bad_history' in df.columns:
        df['Black'] = pd.to_numeric(df['bad_history'], errors='coerce').astype(float)
    elif 'black' in df.columns:
        series = pd.to_numeric(df['black'], errors='coerce')
        uniq = series.dropna().unique()
        if set(uniq).issubset({0, 1}):
            df['Black'] = series.astype(float)
        else:
            df['Black'] = (series > 0.5).astype(float)
    else:
        # if not present, create column with NaNs (will be dropped if required)
        df['Black'] = np.nan

    # Binary controls
    for col in ['self_employed', 'married']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        else:
            df[col] = np.nan

    # Continuous controls that we'll standardize (if present): PI_ratio, loan_to_value, housing_expense_ratio, denied_PMI
    cont_cols = ['PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI']
    for c in cont_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
        else:
            df[c] = np.nan

    # Standardize continuous controls to z-scores (using non-missing mean/std). Create *_z columns.
    for c in cont_cols:
        zname = c + '_z'
        valid = df[c].dropna()
        if len(valid) > 0 and valid.std(ddof=0) != 0:
            mean = valid.mean()
            std = valid.std(ddof=0)
            df[zname] = (df[c] - mean) / std
        else:
            df[zname] = np.nan

    # Prepare final list of columns used in the model
    model_cols = [
        'Approved', 'Female', 'Black', 'PI_ratio_z', 'loan_to_value_z',
        'housing_expense_ratio_z', 'self_employed', 'married', 'denied_PMI_z'
    ]

    # Drop rows missing the DV or IV (these are required). Also drop rows missing all controls used by the model.
    df_model = df[model_cols].copy()

    # For practical modeling, drop rows with NA in Approved or Female, and drop rows that are missing ALL non-binary controls but keep rows if at least some controls present.
    df_model = df_model.dropna(subset=['Approved', 'Female'])

    # For simplicity require at least one of the main continuous controls or binary controls to be present
    required_control_cols = ['Black', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'self_employed', 'married', 'denied_PMI_z']
    df_model = df_model.dropna(subset=required_control_cols, how='all')

    # Cast binary-like columns to numeric (0/1) safe types
    for b in ['Female', 'Black', 'self_employed', 'married']:
        if b in df_model.columns:
            df_model[b] = df_model[b].apply(lambda x: 1.0 if pd.notnull(x) and float(x) >= 0.5 else (0.0 if pd.notnull(x) else np.nan))

    # Return the dataframe with the modeling columns (preserve other original columns if desired by joining) - here we return only the relevant modeling columns for clarity
    return df_model.reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting Approved from Female and the set of controls.
    Returns the fitted statsmodels results object (with robust standard errors).

    Model formula (in matrix form):
      Approved ~ Female + Black + PI_ratio_z + loan_to_value_z + housing_expense_ratio_z + self_employed + married + denied_PMI_z

    The function expects the transformed dataframe produced by transform(...).
    """

    # Ensure required columns exist
    required = ['Approved', 'Female', 'Black', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'self_employed', 'married', 'denied_PMI_z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Drop rows with missing DV or IV
    df_clean = df.dropna(subset=['Approved', 'Female']).copy()

    # For controls, keep rows even if some controls are missing; statsmodels will require complete rows for all included exogs, so drop rows missing any predictor
    predictors = ['Female', 'Black', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'self_employed', 'married', 'denied_PMI_z']
    df_clean = df_clean.dropna(subset=predictors)

    # Prepare design matrix
    X = df_clean[predictors].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df_clean['Approved'].astype(float)

    # Fit binomial GLM (logistic regression) with robust (HC3) standard errors
    model_glm = sm.GLM(y, X, family=sm.families.Binomial())
    results = model_glm.fit()

    # Attach robust covariance (HC3) summary if desired
    try:
        robust_results = results.get_robustcov_results(cov_type='HC3')
    except Exception:
        # fallback to original results if robust fails
        robust_results = results

    # Return the robust results object (caller can print summary)
    return robust_results


