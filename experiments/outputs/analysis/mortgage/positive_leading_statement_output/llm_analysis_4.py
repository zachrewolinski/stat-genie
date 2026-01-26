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
    Transform the raw Boston Fed mortgage dataset to the modeling dataframe.

    Steps:
    - Drop rows with missing values in the variables required for the regression.
    - Ensure numeric types for binary flags and continuous measures.
    - Standardize continuous controls (z-score) so coefficients are comparable.
    - Create an interaction term female_black = female * black to test moderation/intersectionality.

    Returns a dataframe that contains at minimum the columns referenced in the conceptual variables:
    ['female','accept','black','female_black','mortgage_credit_z','consumer_credit_z','PI_ratio_z',
     'loan_to_value_z','housing_expense_ratio_z','bad_history','self_employed','married','denied_PMI']
    """
    df = df.copy()

    # Columns required for analysis
    required = [
        'accept', 'female', 'black',
        'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'bad_history', 'self_employed', 'married', 'denied_PMI'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required)

    # Coerce numeric types for safety
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows made NA by coercion
    df = df.dropna(subset=required)

    # Ensure binary columns are integer 0/1
    for bcol in ['accept', 'female', 'black', 'bad_history', 'self_employed', 'married', 'denied_PMI']:
        df[bcol] = df[bcol].astype(int)

    # Continuous variables to standardize (z-score)
    cont = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # Protect against zero std
        if std == 0 or np.isnan(std):
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Interaction term for intersectional test
    df['female_black'] = df['female'] * df['black']

    # Keep only the columns needed for modeling (but retain original columns as well)
    model_cols = [
        'female', 'accept', 'black', 'female_black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'bad_history', 'self_employed', 'married', 'denied_PMI'
    ]

    # Return a dataframe with the model columns plus any original columns (safer for later inspection)
    return df.loc[:, df.columns.intersection(model_cols).tolist() + [c for c in df.columns if c not in model_cols]]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression for the binary outcome accept using gender (female) as the focal predictor
    and the set of controls supplied in the transformed dataframe.

    Returns a dictionary containing the fitted statsmodels result object and a tidy summary table
    with coefficients, odds ratios, confidence intervals, and p-values.
    """
    # Work on a copy
    data = df.copy()

    # Ensure transform was applied: required model columns
    model_cols = [
        'female', 'black', 'female_black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'bad_history', 'self_employed', 'married', 'denied_PMI',
        'accept'
    ]
    missing = [c for c in model_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Prepare X and y
    X = data[[
        'female', 'black', 'female_black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'bad_history', 'self_employed', 'married', 'denied_PMI'
    ]].astype(float)
    y = data['accept'].astype(int)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression using statsmodels Logit (maximum likelihood)
    # Use try/except to catch potential perfect separation or convergence issues
    try:
        logit_model = sm.Logit(y, X)
        res = logit_model.fit(disp=False)
    except Exception:
        # Fall back to GLM with binomial family (more numerically stable in some cases)
        glm_model = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm_model.fit()

    # Create a tidy summary table with odds ratios and 95% CIs
    params = res.params
    conf = res.conf_int()
    pvalues = res.pvalues

    OR = np.exp(params)
    conf_OR = np.exp(conf)

    summary_table = pd.DataFrame({
        'coef': params,
        'OR': OR,
        'CI_lower': conf_OR[0],
        'CI_upper': conf_OR[1],
        'pvalue': pvalues
    })

    # Order rows with the intercept first
    summary_table = summary_table.loc[summary_table.index]

    return {
        'result': res,
        'summary_table': summary_table
    }


