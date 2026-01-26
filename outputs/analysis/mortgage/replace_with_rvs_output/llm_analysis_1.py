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
    Transform the raw dataset into a cleaned dataframe suitable for logistic regression.
    - Keeps only columns required for the model
    - Drops rows with missing values in those columns
    - Ensures binary indicators are integer-typed
    - Z-scores continuous predictors and appends them with "_z" suffix

    Final dataframe will contain the dependent variable 'accept', the independent variable
    'female', binary controls ('black','bad_history','married','self_employed'), and
    standardized continuous controls: 'mortgage_credit_z','consumer_credit_z',
    'PI_ratio_z','loan_to_value_z','housing_expense_ratio_z'.
    """
    df = df.copy()

    # Columns needed for the model (use original names where available)
    required_cols = [
        'accept',
        'female',
        'black',
        'mortgage_credit',
        'consumer_credit',
        'PI_ratio',
        'loan_to_value',
        'bad_history',
        'married',
        'self_employed',
        'housing_expense_ratio'
    ]

    # Keep only required columns (if any are missing in the dataset this will raise KeyError)
    df = df[required_cols]

    # Drop rows with missing values in any of the required columns
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Ensure binary columns are integer (0/1)
    binary_cols = ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed']
    for col in binary_cols:
        # Some datasets may have floats like 0.0/1.0; cast to int
        df[col] = df[col].astype(int)

    # Continuous predictors to standardize (z-score). Use sample std (ddof=0 or ddof=1 both acceptable).
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in cont_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # Avoid division by zero if std is zero
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Return only the columns used in the model (keeping original binary columns and the standardized continuous ones)
    out_cols = [
        'accept', 'female', 'black', 'bad_history', 'married', 'self_employed',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z'
    ]

    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) to estimate the effect of gender ('female')
    on the probability of mortgage approval ('accept'), controlling for credit- and
    income-related covariates. Returns the fitted model and a table of odds ratios
    with 95% confidence intervals.
    """
    import statsmodels.api as sm

    # Make a copy to avoid modifying input
    data = df.copy()

    # Define predictors (include constant)
    predictors = [
        'female',
        'black',
        'bad_history',
        'married',
        'self_employed',
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z'
    ]

    X = data[predictors]
    X = sm.add_constant(X, has_constant='add')
    y = data['accept']

    # Fit binomial GLM (logistic regression)
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm_binom.fit()

    # Prepare odds ratios and 95% CI
    params = res.params
    conf = res.conf_int()
    or_df = pd.DataFrame({
        'coef': params,
        'odds_ratio': np.exp(params),
        'ci_lower': np.exp(conf[0]),
        'ci_upper': np.exp(conf[1])
    })

    # For clarity, print the summary and the odds ratios table
    print(res.summary())
    print('\nOdds ratios (with 95% CI):')
    print(or_df)

    # Return the fit object and the odds ratio table
    return {
        'model_result': res,
        'odds_ratios': or_df
    }


