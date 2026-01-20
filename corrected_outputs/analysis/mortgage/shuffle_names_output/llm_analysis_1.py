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
    """
    Transform the raw dataframe into the final dataframe for modeling.

    Created columns (and names used in modeling):
    - 'female' : binary indicator 1=female, 0=male (from 'consumer_credit' when present)
    - 'denied' : binary outcome 1=denied, 0=accepted (from 'mortgage_credit')

    Keeps the control variables: 'bad_history', 'loan_to_value', 'PI_ratio',
    'housing_expense_ratio', 'denied_PMI', 'self_employed', 'married'

    Steps:
    - copy dataframe
    - coerce relevant columns to numeric
    - create/standardize the 'female' and 'denied' columns
    - drop rows with missing values in any column required for the model
    """
    df = df.copy()

    # === Create/standardize IV: female ===
    # The schema indicates 'consumer_credit' encodes female (1) vs male (0).
    if 'consumer_credit' in df.columns:
        df['female'] = pd.to_numeric(df['consumer_credit'], errors='coerce')
    else:
        # fallback: if there's already a column named 'female', coerce it
        if 'female' in df.columns:
            df['female'] = pd.to_numeric(df['female'], errors='coerce')
        else:
            # if neither column exists, create NA column to fail later with clear missingness
            df['female'] = np.nan

    # === Create/standardize DV: denied ===
    # According to the schema 'mortgage_credit' is 1 if application was denied, 0 if accepted.
    if 'mortgage_credit' in df.columns:
        df['denied'] = pd.to_numeric(df['mortgage_credit'], errors='coerce')
    elif 'deny' in df.columns:
        # fallback if another column exists that may represent denial
        df['denied'] = pd.to_numeric(df['deny'], errors='coerce')
    else:
        df['denied'] = np.nan

    # === Coerce control variables to numeric (if present) ===
    control_cols = ['bad_history', 'loan_to_value', 'PI_ratio', 'housing_expense_ratio', 'denied_PMI', 'self_employed', 'married']
    for col in control_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # === Drop rows that are missing any of the variables required for the model ===
    required_cols = ['female', 'denied'] + [c for c in control_cols if c in df.columns]
    # ensure uniqueness
    required_cols = list(dict.fromkeys(required_cols))
    df = df.dropna(subset=required_cols)

    # === Cast binary indicators to integers where appropriate ===
    # For safety, any non-binary numeric female/denied values will be thresholded at 0.5
    df['female'] = (df['female'] > 0.5).astype(int)
    df['denied'] = (df['denied'] > 0.5).astype(int)

    # Cast control binaries to int if they appear to be 0/1; leave continuous as numeric
    for bin_col in ['bad_history', 'self_employed', 'married', 'loan_to_value']:
        if bin_col in df.columns:
            # If values are already effectively binary, cast to int
            unique_vals = df[bin_col].dropna().unique()
            if set(np.unique(unique_vals)).issubset({0, 1}):
                df[bin_col] = df[bin_col].astype(int)

    # Return dataframe containing at least the columns used in the model so downstream code can select them
    final_cols = required_cols
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (logit) model to estimate the effect of gender on mortgage denial,
    controlling for applicant and loan characteristics.

    Model specification (in words):
      denied ~ female + bad_history + loan_to_value + PI_ratio + housing_expense_ratio + denied_PMI + self_employed + married

    The function will:
    - select the columns prepared by transform
    - add an intercept
    - fit a statsmodels Logit model
    - return the fitted model object plus odds ratios and 95% confidence intervals
    """
    # Required imports are available in the environment; ensure we have what we need locally
    import statsmodels.api as sm

    # Define the candidate covariate list (in the same order as transform expects)
    covariates = ['female', 'bad_history', 'loan_to_value', 'PI_ratio', 'housing_expense_ratio', 'denied_PMI', 'self_employed', 'married']
    # Keep only covariates that are present in df
    covariates = [c for c in covariates if c in df.columns]

    if 'denied' not in df.columns:
        raise ValueError("Transformed dataframe must contain column 'denied' as the dependent variable.")
    if 'female' not in df.columns:
        raise ValueError("Transformed dataframe must contain column 'female' as the independent variable.")

    X = df[covariates]
    X = sm.add_constant(X, has_constant='add')
    y = df['denied']

    # Fit logistic regression (use LBFGS for stability)
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(method='lbfgs', disp=False)

    # Compute odds ratios and 95% CIs
    params = result.params
    conf = result.conf_int()
    conf.columns = ['2.5%', '97.5%']
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    output = {
        'model_result': result,
        'odds_ratios': odds_ratios,
        'odds_ratio_CI_lower': conf_odds['2.5%'],
        'odds_ratio_CI_upper': conf_odds['97.5%'],
        'covariates_used': X.columns.tolist()
    }

    return output


