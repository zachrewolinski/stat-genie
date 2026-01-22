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
    Prepare the dataset for modeling the effect of gender on mortgage acceptance.

    Steps:
    - Keep only columns needed for the analysis.
    - Drop rows with missing values in any required variable.
    - Create the dependent variable 'Accepted' from 'accept'.
    - Ensure binary columns are integer typed.
    - Standardize continuous / ordinal predictors and place them in new columns prefixed with 's_'.

    Final dataframe contains both original raw control columns (binary) and standardized continuous controls used in the model.
    """
    df = df.copy()

    # Columns we will use (original names from the dataset schema)
    required_cols = [
        'accept', 'female', 'black', 'married', 'self_employed', 'bad_history', 'denied_PMI',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'mortgage_credit', 'consumer_credit',
        'religiousness', 'occupation'
    ]

    # Keep only required columns and drop rows with missing values in any of them
    df = df.loc[:, df.columns.intersection(required_cols)].copy()
    df = df.dropna(subset=required_cols)

    # Dependent variable: Accepted (1 if accepted, 0 if denied)
    df['Accepted'] = df['accept'].astype(int)

    # Independent variable: female (ensure integer 0/1)
    df['female'] = df['female'].astype(int)

    # Binary control columns: ensure integer dtype
    binary_cols = ['black', 'married', 'self_employed', 'bad_history', 'denied_PMI']
    for c in binary_cols:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Continuous / ordinal columns to standardize (create new s_ columns)
    cont_to_standardize = {
        'PI_ratio': 's_PI_ratio',
        'loan_to_value': 's_loan_to_value',
        'housing_expense_ratio': 's_housing_expense_ratio',
        'mortgage_credit': 's_mortgage_credit',
        'consumer_credit': 's_consumer_credit',
        'religiousness': 's_religiousness',
        'occupation': 's_occupation'
    }

    # Standardize (z-score) each continuous/ordinal variable using sample std (ddof=0 for population-style)
    for orig, sname in cont_to_standardize.items():
        if orig in df.columns:
            mean = df[orig].mean()
            std = df[orig].std(ddof=0)
            # avoid division by zero
            if std == 0 or pd.isna(std):
                df[sname] = 0.0
            else:
                df[sname] = (df[orig] - mean) / std

    # Return the transformed dataframe containing all columns required by the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (binomial GLM) predicting mortgage acceptance from applicant gender,
    controlling for applicant financial and demographic characteristics. Returns the fitted model
    and a table of odds ratios with 95% CIs.

    Model specification:
    Accepted ~ female + black + married + self_employed + bad_history + denied_PMI
               + s_PI_ratio + s_loan_to_value + s_housing_expense_ratio
               + s_mortgage_credit + s_consumer_credit + s_religiousness + s_occupation

    Uses statsmodels Logit for maximum likelihood estimation.
    """
    import statsmodels.api as sm
    import numpy as np
    import pandas as pd

    df = df.copy()

    # Define the columns used in the model (must match transform output)
    feature_cols = [
        'female',
        'black', 'married', 'self_employed', 'bad_history', 'denied_PMI',
        's_PI_ratio', 's_loan_to_value', 's_housing_expense_ratio',
        's_mortgage_credit', 's_consumer_credit', 's_religiousness', 's_occupation'
    ]

    # Ensure all features are present
    missing = [c for c in feature_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Drop any remaining rows with NA in the model columns (defensive)
    model_df = df[[ 'Accepted' ] + feature_cols].dropna()

    y = model_df['Accepted'].astype(int)
    X = model_df[feature_cols]
    X = sm.add_constant(X)

    # Fit logistic regression (Logit). Use disp=False to suppress optimization output.
    logit_res = sm.Logit(y, X).fit(disp=False)

    # Compute odds ratios and 95% confidence intervals
    params = logit_res.params
    conf = logit_res.conf_int()
    or_df = pd.DataFrame({
        'OR': np.exp(params),
        'CI_lower': np.exp(conf[0]),
        'CI_upper': np.exp(conf[1])
    })

    # Return result objects: the fitted model and the odds-ratio table
    return {
        'model_result': logit_res,
        'odds_ratios': or_df
    }


