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
    Clean and prepare the Boston mortgage dataset for logistic regression.
    - Keep only columns needed for modeling.
    - Coerce to numeric and drop rows with missing values in those columns.
    - Create standardized (z-scored) versions of continuous predictors for stability and interpretability.

    Final dataframe will include
      - accept (DV), female (IV), binary controls (black, bad_history, self_employed, married, denied_PMI)
      - standardized continuous controls: mortgage_credit_z, consumer_credit_z, PI_ratio_z, loan_to_value_z, housing_expense_ratio_z
    """
    df = df.copy()

    # Columns required for the analysis
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit', 'PI_ratio',
        'loan_to_value', 'bad_history', 'housing_expense_ratio', 'self_employed', 'married', 'denied_PMI'
    ]

    # Ensure columns exist (if a few are missing, this will raise a clear KeyError)
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Keep only the required columns to simplify downstream analysis
    df = df[required_cols].copy()

    # Coerce to numeric and set NaN for non-convertible values
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any of the required modeling columns
    df = df.dropna(subset=required_cols)

    # Ensure binary indicators are integers (0/1)
    binary_cols = ['accept', 'female', 'black', 'bad_history', 'self_employed', 'married', 'denied_PMI']
    for c in binary_cols:
        df[c] = df[c].astype(int)

    # Standardize continuous predictors (use population std ddof=0 for stability)
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If zero variance (unlikely), create zero column to avoid division by zero
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Return the transformed dataframe with all columns necessary for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting acceptance of mortgage (accept) as a function of
    applicant gender (female) and controls. Returns the fitted statsmodels result and a
    table of odds ratios with 95% confidence intervals and p-values.

    Model formula:
      accept ~ female + black + mortgage_credit_z + consumer_credit_z + PI_ratio_z
               + loan_to_value_z + bad_history + housing_expense_ratio_z
               + self_employed + married + denied_PMI
    """
    import statsmodels.formula.api as smf

    # Ensure required transformed columns exist
    required_model_cols = [
        'accept', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z',
        'loan_to_value_z', 'bad_history', 'housing_expense_ratio_z', 'self_employed', 'married', 'denied_PMI'
    ]
    missing = [c for c in required_model_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for model: {missing}")

    formula = (
        'accept ~ female + black + mortgage_credit_z + consumer_credit_z + PI_ratio_z '
        '+ loan_to_value_z + bad_history + housing_expense_ratio_z + self_employed + married + denied_PMI'
    )

    # Fit logistic regression
    result = smf.logit(formula=formula, data=df).fit(disp=False)

    # Compute odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    conf.columns = ['2.5%', '97.5%']
    odds = np.exp(params)
    conf_exp = np.exp(conf)

    or_table = pd.DataFrame({
        'OR': odds,
        'OR_2.5%': conf_exp['2.5%'],
        'OR_97.5%': conf_exp['97.5%'],
        'pvalue': result.pvalues
    })

    # Return a dictionary with the fitted result object and odds ratio table
    results = {
        'fitted_model': result,
        'odds_ratios': or_table
    }
    return results


