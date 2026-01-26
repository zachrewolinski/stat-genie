from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/add_features_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston Fed mortgage dataset into the modeling dataframe.

    Steps:
    - Drop rows with missing values in variables required for the analysis.
    - Ensure binary indicators are integer typed.
    - Standardize continuous control variables (z-scores) to help numeric stability and interpretation.
    - Return a dataframe that contains the dependent variable (accept), independent variable (female),
      and all control columns listed in the conceptual variables with exact column names.
    """
    df = df.copy()

    # Columns required for the analysis
    required_cols = [
        'female', 'accept', 'black', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'self_employed', 'married',
        'occupation', 'persons'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Ensure binary variables are integers (0/1)
    df['female'] = df['female'].astype(int)
    df['accept'] = df['accept'].astype(int)
    df['black'] = df['black'].astype(int)
    df['bad_history'] = df['bad_history'].astype(int)
    df['self_employed'] = df['self_employed'].astype(int)
    df['married'] = df['married'].astype(int)

    # Continuous/ordinal variables to standardize (z-score)
    cont_cols = [
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value',
        'housing_expense_ratio', 'occupation', 'persons'
    ]

    # Compute population z-score (ddof=0) for each continuous control and store with _z suffix
    for c in cont_cols:
        col_mean = df[c].mean()
        col_std = df[c].std(ddof=0)
        # If std is zero (unlikely), set z to zero to avoid division by zero
        if col_std == 0 or np.isnan(col_std):
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - col_mean) / col_std

    # Select and return only the columns we will use in the model (exact names)
    out_cols = [
        'female', 'accept', 'black', 'mortgage_credit_z', 'consumer_credit_z', 'bad_history',
        'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'self_employed', 'married',
        'occupation_z', 'persons_z'
    ]

    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression to estimate the effect of gender (female) on mortgage acceptance
    while controlling for applicant creditworthiness and other covariates.

    Model: logit( P(accept=1) ) = beta0 + beta1*female + sum_k gamma_k * control_k

    Returns a dictionary with the fitted model object, textual summary, and a DataFrame of
    odds ratios (OR), 95% CI for OR, and p-values.
    """
    df = df.copy()

    # Predictor columns (must match the transform output)
    X_cols = [
        'female', 'black', 'mortgage_credit_z', 'consumer_credit_z', 'bad_history',
        'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'self_employed', 'married',
        'occupation_z', 'persons_z'
    ]

    # Prepare design matrix and response
    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (use Logit; fallback to GLM if convergence/separation issues arise)
    try:
        fit_res = sm.Logit(y, X).fit(disp=False)
    except Exception as e:
        # fallback to GLM (binomial) which sometimes handles separation differently
        fit_res = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Extract parameter estimates, confidence intervals, p-values, and compute odds ratios
    params = fit_res.params
    conf = fit_res.conf_int()
    pvalues = fit_res.pvalues

    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    odds_df = pd.DataFrame({
        'OR': odds_ratios,
        'CI_lower': conf_odds[0],
        'CI_upper': conf_odds[1],
        'pvalue': pvalues
    })

    results = {
        'model': fit_res,
        'summary': fit_res.summary().as_text(),
        'odds_ratios': odds_df
    }

    return results


