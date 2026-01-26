from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/replace_with_rvs_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.
    Steps:
    - Make a defensive copy
    - Ensure relevant columns are numeric
    - Drop rows with missing DV/IV or core control variables
    - Create standardized versions of continuous controls (mean 0, sd 1)
    - Return dataframe containing original variables plus standardized columns
    The final dataframe will contain the columns referenced in the conceptual variables:
      - 'accept' (DV), 'female' (IV)
      - controls: 'black','self_employed','married','bad_history','denied_PMI',
        'mortgage_credit_std','consumer_credit_std','PI_ratio_std','housing_expense_ratio_std','loan_to_value_std'
    """
    df = df.copy()

    # Columns expected in the raw data (based on provided schema)
    expected_cols = [
        'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
        'deny', 'loan_to_value', 'denied_PMI', 'accept'
    ]

    # Coerce to numeric where appropriate (safely convert strings that represent numbers)
    for c in expected_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing DV or IV
    df = df.dropna(subset=['accept', 'female'])

    # Core control variables we will require for the multivariate model
    required_controls = [
        'black', 'mortgage_credit', 'consumer_credit', 'PI_ratio',
        'loan_to_value', 'housing_expense_ratio', 'self_employed',
        'married', 'bad_history', 'denied_PMI'
    ]

    # Keep only rows that have non-missing values for the required controls
    present_required = [c for c in required_controls if c in df.columns]
    if present_required:
        df = df.dropna(subset=present_required)

    # Standardize continuous controls (create new columns with _std suffix)
    continuous = [c for c in ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio'] if c in df.columns]
    for c in continuous:
        # Use population std (ddof=0) for stable scaling; if std is zero, fill with zeros
        std = df[c].std(ddof=0)
        mean = df[c].mean()
        new_col = c + '_std'
        if std == 0 or np.isclose(std, 0):
            df[new_col] = 0.0
        else:
            df[new_col] = (df[c] - mean) / std

    # Ensure binary controls are 0/1 integers
    binary_controls = [c for c in ['female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI', 'accept'] if c in df.columns]
    for c in binary_controls:
        df[c] = df[c].astype(float).round(0).astype(int)

    # Final check: keep only rows where DV is 0/1
    df = df[df['accept'].isin([0, 1])]

    # Return transformed dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression to estimate the effect of gender on mortgage acceptance,
    controlling for applicant creditworthiness and other underwriting factors.

    Returns a dictionary with:
      - 'model_results': the fitted statsmodels result object
      - 'odds_ratios': series of odds ratios for each coefficient
      - 'odds_ratio_conf_int': exponentiated confidence intervals for odds ratios
      - 'marginal_effects': DataFrame with average marginal effects (if available)
    """
    df = df.copy()

    # Columns to include as predictors (these must exist in the transformed df)
    predictor_cols = [
        'female',
        'black',
        'self_employed',
        'married',
        'bad_history',
        'denied_PMI'
    ]

    # Add standardized continuous predictors if present
    for cont in ['mortgage_credit_std', 'consumer_credit_std', 'PI_ratio_std', 'housing_expense_ratio_std', 'loan_to_value_std']:
        if cont in df.columns:
            predictor_cols.append(cont)

    # Ensure predictors exist in df
    predictor_cols = [c for c in predictor_cols if c in df.columns]

    if 'accept' not in df.columns:
        raise ValueError("Transformed dataframe must contain 'accept' column as dependent variable.")

    X = df[predictor_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression; fallback to GLM binomial if convergence problems
    try:
        logit_mod = sm.Logit(y, X)
        res = logit_mod.fit(disp=False, maxiter=200)
    except Exception:
        glm_mod = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm_mod.fit()

    # Compute odds ratios and exponentiated confidence intervals
    params = res.params
    odds_ratios = np.exp(params)
    try:
        conf = res.conf_int()
        conf_odds = np.exp(conf)
    except Exception:
        conf_odds = None

    # Compute average marginal effects if supported
    try:
        margeff = res.get_margeff(at='overall')
        marg_df = margeff.summary_frame()
    except Exception:
        marg_df = None

    return {
        'model_results': res,
        'odds_ratios': odds_ratios,
        'odds_ratio_conf_int': conf_odds,
        'marginal_effects': marg_df
    }


