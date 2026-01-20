from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw mortgage dataset for modeling the effect of gender on approval.

    Outputs (added/renamed columns used in modeling):
    - Approved: binary DV (from 'accept').
    - female: IV (kept as-is, ensure int 0/1).
    - binary controls (kept as ints): black, self_employed, married, bad_history, denied_PMI
    - standardized continuous controls: housing_expense_ratio_z, mortgage_credit_z, consumer_credit_z, PI_ratio_z, loan_to_value_z
    
    The function drops rows with missing values in any of the columns used in the model.
    """
    df = df.copy()

    # Create DV: ensure binary int 0/1
    if 'accept' not in df.columns:
        raise KeyError("Input dataframe must contain 'accept' column")
    df['Approved'] = df['accept'].astype(int)

    # Define the columns we will use in the model
    continuous_cols = ['housing_expense_ratio', 'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value']
    binary_cols = ['female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']

    required_cols = ['Approved'] + continuous_cols + binary_cols

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are integers (0/1)
    for c in binary_cols:
        # If column exists, coerce to int; otherwise raise
        if c not in df.columns:
            raise KeyError(f"Input dataframe must contain '{c}' column")
        df[c] = df[c].astype(int)

    # Standardize continuous controls (z-score) and create new columns with suffix '_z'
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit-transform on the available rows (after NA drop)
    df_cont = df[continuous_cols].astype(float)
    cont_z = scaler.fit_transform(df_cont)
    cont_z_cols = [c + '_z' for c in continuous_cols]
    df[cont_z_cols] = cont_z

    # Return dataframe containing at least the columns used in the model
    # (We keep other columns too; but ensure required columns are present.)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (logit) predicting approval from gender and controls.

    Model specification (log-odds of approval):
      Approved ~ female + black + self_employed + married + bad_history + denied_PMI
                 + housing_expense_ratio_z + mortgage_credit_z + consumer_credit_z + PI_ratio_z + loan_to_value_z

    Returns the fitted statsmodels results object (Logit fitted model).
    """
    import statsmodels.api as sm

    # Ensure the transformed columns exist
    required_model_cols = [
        'Approved', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'housing_expense_ratio_z', 'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z'
    ]
    missing = [c for c in required_model_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    df_model = df.copy()
    y = df_model['Approved'].astype(int)

    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'housing_expense_ratio_z', 'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z'
    ]
    X = df_model[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (use GLM with binomial family or Logit)
    # Using sm.Logit for likelihood-based logit estimation
    logit = sm.Logit(y, X)
    # Fit quietly
    try:
        res = logit.fit(disp=False)
    except Exception:
        # fallback to GLM if Logit has convergence problems
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    return res


