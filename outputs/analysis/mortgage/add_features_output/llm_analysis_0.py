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
    Transform the raw Boston Fed mortgage dataset into a dataframe ready for modeling.

    Produces/keeps the following columns (used in the model):
      - accept (int: 1 accepted, 0 denied)
      - female (int: 1 female, 0 male)
      - black (int: 1 Black, 0 otherwise)
      - mortgage_credit_z, consumer_credit_z, PI_ratio_z, loan_to_value_z, housing_expense_ratio_z (standardized continuous controls)
      - bad_history, married, self_employed (binary controls)

    The function drops rows with missing values in any of these columns.
    """
    # Select columns needed for analysis
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'bad_history', 'married', 'self_employed'
    ]

    # Keep only required columns (if some not present, raise an informative error)
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Make a working copy to avoid changing the caller's dataframe
    df = df.copy()

    # Convert binary columns to integer (safe coercion)
    for bcol in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed']:
        df[bcol] = pd.to_numeric(df[bcol], errors='coerce').astype('Int64')

    # Convert continuous columns to numeric
    for ccol in ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']:
        df[ccol] = pd.to_numeric(df[ccol], errors='coerce')

    # Drop rows with missing values in any variable we plan to use
    df = df.dropna(subset=required_cols)

    # Now coerce binary columns to plain int (0/1)
    for bcol in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed']:
        df[bcol] = df[bcol].astype(int)

    # Standardize continuous controls (z-score). Use population std (ddof=0) to match many modeling conventions.
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for ccol in cont_cols:
        mean = df[ccol].mean()
        std = df[ccol].std(ddof=0)
        # If std is zero (unlikely), produce zeros to avoid division by zero
        if std == 0 or np.isnan(std):
            df[ccol + '_z'] = 0.0
        else:
            df[ccol + '_z'] = (df[ccol] - mean) / std

    # Retain only the final columns used in the model to make modeling function straightforward
    final_cols = [
        'accept', 'female', 'black', 'mortgage_credit_z', 'consumer_credit_z',
        'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'bad_history', 'married', 'self_employed'
    ]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting loan acceptance from applicant gender
    controlling for key financial and demographic covariates.

    Returns a dictionary with:
      - 'model': fitted statsmodels Logit results object
      - 'summary': the model summary (string)
      - 'odds_ratio': DataFrame with coef, odds_ratio, 95% CI, and p-values

    Model specification:
      accept ~ female + black + bad_history + married + self_employed
               + mortgage_credit_z + consumer_credit_z + PI_ratio_z + loan_to_value_z + housing_expense_ratio_z

    Robustness: uses default Logit fit. Users may request clustered or robust SEs separately.
    """
    # Ensure required columns are present
    model_cols = [
        'female', 'black', 'bad_history', 'married', 'self_employed',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z'
    ]
    missing = [c for c in model_cols + ['accept'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Prepare X and y
    X = df[model_cols].astype(float)
    X = sm.add_constant(X)
    y = df['accept'].astype(int)

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    try:
        res = logit_model.fit(disp=False)
    except Exception as e:
        # If convergence fails, try a GLM Binomial (more stable) as a fallback
        glm_model = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm_model.fit()

    # Compute odds ratios and 95% confidence intervals
    params = res.params
    conf = res.conf_int()
    or_df = pd.DataFrame({
        'coef': params,
        'odds_ratio': np.exp(params),
        'ci_lower': np.exp(conf[0]),
        'ci_upper': np.exp(conf[1]),
        'pvalue': res.pvalues
    })

    results = {
        'model': res,
        'summary': res.summary().as_text(),
        'odds_ratio': or_df
    }

    return results


