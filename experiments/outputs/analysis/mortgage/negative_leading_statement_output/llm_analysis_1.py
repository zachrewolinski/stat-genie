from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/negative_leading_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for modeling. Steps:
    - Copy input dataframe.
    - Drop rows with missing values in variables required for the analysis.
    - Ensure binary variables are integer-coded (0/1).
    - Standardize numeric credit/risk covariates to z-scores for stable estimation.
    - Return dataframe containing all columns referenced in the conceptual model.
    """
    df = df.copy()

    # Required columns for the analysis
    required_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio'
    ]

    # Drop rows missing any required variable
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are integers (0/1)
    binary_cols = ['accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        # safe cast: if values are floats but represent 0/1
        df[c] = df[c].astype(int)

    # Standardize continuous/ordinal numeric covariates (z-score)
    to_scale = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in to_scale:
        # if constant (std==0) avoid division by zero
        std = df[c].std()
        mean = df[c].mean()
        if std == 0 or np.isclose(std, 0):
            # create zero column if no variation
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Keep only columns we need for modeling (but keep others if desired)
    model_cols = [
        'accept', 'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z'
    ]

    # Some dataframes may contain those columns already; ensure they exist
    missing_model_cols = [c for c in model_cols if c not in df.columns]
    if missing_model_cols:
        raise ValueError(f"Missing required transformed columns: {missing_model_cols}")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression of loan acceptance on applicant female indicator (primary IV)
    and the set of control variables that capture creditworthiness and demographics.

    Returns a dictionary containing:
      - 'model': the fitted statsmodels LogitResults object
      - 'odds_ratios': exponentiated coefficients (odds ratios)
      - 'conf_int_odds': 95% CI for odds ratios
      - 'pvalues': p-values for coefficients

    The primary test of interest is the coefficient on 'female'. A statistically significant
    coefficient (p < .05) would suggest gender is associated with approval probability
    after controlling for the included covariates. If non-significant and effect size small,
    that provides evidence consistent with the hypothesis that gender does not affect approval.
    """
    df = df.copy()

    # Define predictors: female plus controls (already standardized where appropriate)
    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z'
    ]

    # Confirm columns are present
    missing = [c for c in X_cols + ['accept'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['accept']

    # Fit logistic regression (use GLM with Binomial or Logit). Using Logit here.
    try:
        logit_model = sm.Logit(y, X)
        res = logit_model.fit(disp=False, method='lbfgs')
    except Exception:
        # fallback to GLM if Logit has convergence problems
        glm_model = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm_model.fit()

    # Odds ratios and 95% confidence intervals for odds ratios
    params = res.params
    conf = res.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    results = {
        'model': res,
        'odds_ratios': odds_ratios,
        'conf_int_odds': conf_odds,
        'pvalues': res.pvalues
    }

    return results


