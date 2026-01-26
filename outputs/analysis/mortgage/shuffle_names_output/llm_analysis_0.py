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
    Clean and prepare the mortgage dataset for modeling.

    Produces a dataframe with the exact columns used in the model:
      - Approved: 1 if application accepted, 0 if denied
      - Female: 1 if applicant is female, 0 if male
      - PI_ratio, loan_to_value, housing_expense_ratio, self_employed, married, bad_history

    The function attempts to be robust to small variations in the raw column names described in the schema.
    """
    df = df.copy()

    # --- Determine gender column (prefer 'consumer_credit' per schema; fallback to 'female') ---
    if 'consumer_credit' in df.columns:
        df['Female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').astype(float)
    elif 'female' in df.columns:
        # fallback (some files have a 'female' column) -- coerce to 0/1 if not already
        df['Female'] = pd.to_numeric(df['female'], errors='coerce').astype(float)
    else:
        raise KeyError("No gender column found. Expected 'consumer_credit' or 'female'.")

    # Ensure binary 0/1 (if values are probabilities or slight floats, threshold at 0.5)
    df['Female'] = df['Female'].map(lambda x: np.nan if pd.isna(x) else (1 if x >= 0.5 else 0)).astype('float')

    # --- Determine approval/denial outcome and create Approved indicator ---
    # Prefer 'mortgage_credit' which the schema indicates: 1 = denied, 0 = accepted
    if 'mortgage_credit' in df.columns:
        df['Approved'] = pd.to_numeric(df['mortgage_credit'], errors='coerce').map(lambda x: 1 if x == 0 else (0 if x == 1 else np.nan)).astype('float')
    elif 'Unnamed: 0' in df.columns:
        # schema notes this may be 1 if accepted, 0 if denied
        df['Approved'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce').astype(float)
    elif 'deny' in df.columns:
        # 'deny' may be counts or another encoding; if it's binary with 1=denied 0=accepted, convert
        # But because schema is ambiguous, only convert if values are {0,1}
        tmp = pd.to_numeric(df['deny'], errors='coerce')
        uniq = set(tmp.dropna().unique())
        if uniq.issubset({0, 1}):
            df['Approved'] = tmp.map(lambda x: 1 if x == 0 else 0).astype('float')
        else:
            raise KeyError("Found 'deny' but values are not binary 0/1; cannot reliably construct Approved indicator.")
    else:
        raise KeyError("No suitable approval/denial column found. Expected 'mortgage_credit' or 'Unnamed: 0' or binary 'deny'.")

    # --- Pull control variables; if missing create as NaN (later we'll drop rows missing required fields) ---
    control_cols = ['PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'self_employed', 'married', 'bad_history']
    for col in control_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # create the column with NaNs so the final dataframe has consistent columns
            df[col] = np.nan

    # --- Final selection and row filtering ---
    model_cols = ['Approved', 'Female'] + control_cols
    # Drop rows with missing outcome or IV; also drop rows missing all controls would be harmful, so require at least the main controls to be present
    # Here we require Approved and Female, and at least PI_ratio and loan_to_value to be non-missing (these are primary credit controls).
    required_subset = ['Approved', 'Female', 'PI_ratio', 'loan_to_value']
    df_final = df.dropna(subset=required_subset).copy()

    # Cast binary control columns to 0/1 where appropriate
    for bcol in ['self_employed', 'married', 'bad_history']:
        if bcol in df_final.columns:
            # map values of >0.5 to 1, <=0.5 to 0
            df_final[bcol] = df_final[bcol].map(lambda x: np.nan if pd.isna(x) else (1 if x >= 0.5 else 0)).astype('float')

    # Keep only the final model columns in the returned dataframe (in the exact order expected by model)
    df_final = df_final[model_cols]

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression to estimate the effect of gender (Female) on mortgage approval (Approved),
    controlling for standard credit and applicant characteristics.

    Returns a dictionary with the fitted model object and a small summary table of odds ratios.
    """
    # Ensure input contains the expected columns
    expected = ['Approved', 'Female', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'self_employed', 'married', 'bad_history']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for modeling: {missing}")

    # Drop any remaining rows with missing values in modelling columns
    df_model = df.dropna(subset=expected).copy()

    # Prepare design matrix
    X = df_model[['Female', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'self_employed', 'married', 'bad_history']].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df_model['Approved'].astype(float)

    # Fit logistic regression (maximum likelihood)
    # Use robust (HC3) standard errors when reporting if desired
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)

    # Compute odds ratios and 95% CI for easy interpretation
    params = res.params
    conf = res.conf_int()
    odds_ratios = pd.DataFrame({
        'OR': np.exp(params),
        'CI_lower': np.exp(conf[0]),
        'CI_upper': np.exp(conf[1])
    })

    # Package results
    results = {
        'model_result': res,            # statsmodels fitted result object
        'odds_ratios': odds_ratios,     # DataFrame of ORs and CIs
        'n_obs': int(res.nobs),
        'aic': float(res.aic)
    }

    return results


