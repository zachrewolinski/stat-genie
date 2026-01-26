from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/positive_leading_statement_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and standardize columns required for modeling whether gender affects mortgage approval.

    Inputs: raw dataframe with columns including at least the following:
      ['female','black','housing_expense_ratio','self_employed','married',
       'mortgage_credit','consumer_credit','bad_history','PI_ratio',
       'loan_to_value','denied_PMI','accept']

    Outputs: dataframe containing the following columns used by the model:
      - Accepted (DV)
      - Female (IV)
      - Black, SelfEmployed, Married, BadHistory, Denied_PMI (binary controls)
      - HousingExpenseRatio_z, PI_Ratio_z, LoanToValue_z, MortgageCredit_z, ConsumerCredit_z (standardized continuous controls)
    """
    df = df.copy()

    # Columns required (as present in the dataset schema)
    required = [
        'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
        'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
        'loan_to_value', 'denied_PMI', 'accept'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required)

    # Create clear column names used in modeling
    df['Accepted'] = df['accept'].astype(int)
    df['Female'] = df['female'].astype(int)
    df['Black'] = df['black'].astype(int)
    df['HousingExpenseRatio'] = df['housing_expense_ratio'].astype(float)
    df['SelfEmployed'] = df['self_employed'].astype(int)
    df['Married'] = df['married'].astype(int)
    df['MortgageCredit'] = df['mortgage_credit'].astype(float)
    df['ConsumerCredit'] = df['consumer_credit'].astype(float)
    df['BadHistory'] = df['bad_history'].astype(int)
    df['PI_Ratio'] = df['PI_ratio'].astype(float)
    df['LoanToValue'] = df['loan_to_value'].astype(float)
    df['Denied_PMI'] = df['denied_PMI'].astype(int)

    # Standardize continuous / ordinal predictors for numerical stability and interpretability
    to_standardize = ['HousingExpenseRatio', 'PI_Ratio', 'LoanToValue', 'MortgageCredit', 'ConsumerCredit']
    for col in to_standardize:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # avoid division by zero; if std==0 produce zero column
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Keep only the columns we will use downstream (plus originals for traceability)
    keep_cols = [
        'Accepted', 'Female', 'Black', 'HousingExpenseRatio_z', 'SelfEmployed', 'Married',
        'MortgageCredit_z', 'ConsumerCredit_z', 'BadHistory', 'PI_Ratio_z', 'LoanToValue_z', 'Denied_PMI'
    ]

    # Ensure columns exist (in case some standardized columns were filled with zeros above)
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression to estimate the effect of applicant gender (Female) on mortgage acceptance
    while controlling for observed creditworthiness and application-related covariates.

    Returns a dictionary containing:
      - 'model': fitted statsmodels Logit result object
      - 'summary_table': DataFrame with coefficients, odds ratios, 95% CIs, and p-values
      - 'female_margeff': average marginal effect for Female (if computation succeeds)
    """
    # Required libraries already imported at top-level (np, pd, sm)

    # Define the feature list used in the model (matches the transformed dataframe)
    features = [
        'Female', 'Black', 'HousingExpenseRatio_z', 'SelfEmployed', 'Married',
        'MortgageCredit_z', 'ConsumerCredit_z', 'BadHistory', 'PI_Ratio_z', 'LoanToValue_z', 'Denied_PMI'
    ]

    # Drop rows with missing values in features or outcome
    df_model = df.dropna(subset=['Accepted'] + features).copy()

    # Build design matrices
    X = sm.add_constant(df_model[features], has_constant='add')
    y = df_model['Accepted'].astype(int)

    # Fit logistic regression (MLE)
    # Use try/except to handle potential convergence/separation issues gracefully
    try:
        logit = sm.Logit(y, X)
        fit = logit.fit(disp=False)
    except Exception as e:
        # If Logit fails (e.g., perfect separation), fall back to GLM with binomial family
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        fit = glm.fit()

    # Create a summary table with odds ratios and confidence intervals
    params = fit.params
    pvals = fit.pvalues
    conf = fit.conf_int()
    conf.columns = ['ci_lower', 'ci_upper']

    summary_table = pd.DataFrame({
        'coef': params,
        'pvalue': pvals,
        'ci_lower': conf['ci_lower'],
        'ci_upper': conf['ci_upper']
    })
    summary_table['odds_ratio'] = np.exp(summary_table['coef'])
    summary_table['odds_ratio_ci_lower'] = np.exp(summary_table['ci_lower'])
    summary_table['odds_ratio_ci_upper'] = np.exp(summary_table['ci_upper'])

    # Compute average marginal effect (AME) for Female if available
    female_margeff = None
    try:
        margeff_obj = fit.get_margeff(at='overall', method='dydx')
        # margeff_obj.margeff is array aligned with features (and const). Find Female's effect
        # The returned object contains .summary() for display; we return the object itself for inspection
        female_margeff = margeff_obj
    except Exception:
        # If get_margeff not available for the fit object, keep None
        female_margeff = None

    results = {
        'model': fit,
        'summary_table': summary_table,
        'female_margeff': female_margeff
    }

    return results


