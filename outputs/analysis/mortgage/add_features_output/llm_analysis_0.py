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
    Transform the raw Boston Fed mortgage dataset for modeling the effect of gender on mortgage acceptance.

    Outputs a dataframe with the following relevant columns (names used by the model):
      - accept (DV, binary)
      - female (IV, binary)
      - black (control & moderator, binary)
      - mortgage_credit_z, consumer_credit_z, PI_ratio_z, loan_to_value_z (standardized continuous controls)
      - bad_history, married, self_employed (binary controls)

    Rows with missing values on any variables required for the model are dropped.
    """
    df = df.copy()

    # Ensure key columns exist
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'PI_ratio', 'loan_to_value', 'bad_history', 'married', 'self_employed'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Required columns missing from dataframe: {missing}")

    # Keep only rows with non-missing values in required columns
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are numeric integers (0/1)
    for bin_col in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed']:
        # coerce to numeric then to 0/1 where possible
        df[bin_col] = pd.to_numeric(df[bin_col], errors='coerce')
        # After numeric coercion drop rows with NaN
    df = df.dropna(subset=['accept', 'female', 'black', 'bad_history', 'married', 'self_employed'])

    df['accept'] = df['accept'].astype(int)
    df['female'] = df['female'].astype(int)
    df['black'] = df['black'].astype(int)
    df['bad_history'] = df['bad_history'].astype(int)
    df['married'] = df['married'].astype(int)
    df['self_employed'] = df['self_employed'].astype(int)

    # Standardize continuous / ordinal covariates for more stable estimation
    cont_vars = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value']
    for col in cont_vars:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=cont_vars)

    # Compute z-scores (sample std, ddof=1) and store new columns with exact names used in model
    df['mortgage_credit_z'] = (df['mortgage_credit'] - df['mortgage_credit'].mean()) / (df['mortgage_credit'].std(ddof=1) if df['mortgage_credit'].std(ddof=1) != 0 else 1)
    df['consumer_credit_z'] = (df['consumer_credit'] - df['consumer_credit'].mean()) / (df['consumer_credit'].std(ddof=1) if df['consumer_credit'].std(ddof=1) != 0 else 1)
    df['PI_ratio_z'] = (df['PI_ratio'] - df['PI_ratio'].mean()) / (df['PI_ratio'].std(ddof=1) if df['PI_ratio'].std(ddof=1) != 0 else 1)
    df['loan_to_value_z'] = (df['loan_to_value'] - df['loan_to_value'].mean()) / (df['loan_to_value'].std(ddof=1) if df['loan_to_value'].std(ddof=1) != 0 else 1)

    # Keep only columns necessary for modeling + originals for reference
    model_cols = [
        'accept', 'female', 'black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z',
        'bad_history', 'married', 'self_employed'
    ]

    # Final drop of any remaining missing values in model columns
    df = df.dropna(subset=model_cols)

    # Return the dataframe containing all columns (including model_cols). The model function will use these columns.
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (GLM with binomial family) predicting mortgage acceptance.

    Model formula includes the main effect of female, main effect of black, their interaction,
    and a set of controls for creditworthiness and loan risk. The function returns the fitted model results object.

    Formula used:
      accept ~ female + black + female:black
               + mortgage_credit_z + consumer_credit_z + PI_ratio_z + loan_to_value_z
               + bad_history + married + self_employed

    Interpretation: coefficient on 'female' is the log-odds difference in acceptance for female vs male
    among non-Black applicants (because we include female:black interaction). The interaction term
    tests whether the gender effect differs for Black applicants.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns are present
    required = [
        'accept', 'female', 'black',
        'mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z',
        'bad_history', 'married', 'self_employed'
    ]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing columns required for modeling: {missing}")

    formula = (
        'accept ~ female + black + female:black '
        '+ mortgage_credit_z + consumer_credit_z + PI_ratio_z + loan_to_value_z '
        '+ bad_history + married + self_employed'
    )

    # Fit GLM (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = model.fit()

    # Return the fitted results object for downstream inspection (summary(), params, conf_int(), etc.)
    return results

