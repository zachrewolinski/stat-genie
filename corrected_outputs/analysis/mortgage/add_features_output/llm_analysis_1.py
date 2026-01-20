from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/add_features_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston Fed mortgage dataset to a modeling-ready dataframe.

    Produces the following columns (exact names used in the model):
      - Accepted: binary outcome (1 = accepted, 0 = denied) from 'accept'
      - female: original binary gender indicator (1 = female, 0 = male)
      - female_black_interaction: interaction female * black
      - black: original binary race indicator (1 = Black)
      - mortgage_credit_z, consumer_credit_z, PI_ratio_z, loan_to_value_z, housing_expense_ratio_z: z-scored continuous controls
      - bad_history, self_employed, married: binary controls

    Rows with missing values in any of the variables used in the model are dropped.
    """
    # Work on a copy
    df = df.copy()

    # Columns required for the model
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'bad_history', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio',
        'self_employed', 'married'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Outcome: Accepted (1 = accepted, 0 = denied)
    # Use 'accept' column as provided (1 = accepted, 0 = denied)
    df['Accepted'] = df['accept'].astype(int)

    # Ensure binary indicators are integers (0/1)
    binary_cols = ['female', 'black', 'bad_history', 'self_employed', 'married']
    for c in binary_cols:
        # convert to numeric and then int (safe guard if floats)
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(int)

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for stability.
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for c in cont_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        # If std is zero (unlikely), avoid division by zero
        if std == 0 or np.isnan(std):
            df[c + '_z'] = df[c] - mean
        else:
            df[c + '_z'] = (df[c] - mean) / std

    # Interaction term: female * black (to test moderation of gender effect by race)
    df['female_black_interaction'] = df['female'] * df['black']

    # Final dataframe with only the columns necessary for modeling (keeps order explicit)
    final_cols = [
        'Accepted',
        'female',
        'female_black_interaction',
        'black',
        'mortgage_credit_z',
        'consumer_credit_z',
        'bad_history',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z',
        'self_employed',
        'married'
    ]

    # Some safety: ensure final columns exist (in case any earlier transform created NaNs)
    df = df.dropna(subset=final_cols)

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (maximum likelihood) predicting mortgage acceptance from gender
    while controlling for creditworthiness and demographic covariates. Includes a female x black
    interaction to test whether the gender effect differs by race.

    Model specification (logit):
      Accepted ~ female + female_black_interaction + black
                 + mortgage_credit_z + consumer_credit_z + bad_history
                 + PI_ratio_z + loan_to_value_z + housing_expense_ratio_z
                 + self_employed + married

    Returns the fitted statsmodels result object.
    """
    # Work on a copy
    df = df.copy()

    # Define outcome and predictors exactly as in the transformed dataframe
    y = df['Accepted']
    X = df[[
        'female',
        'female_black_interaction',
        'black',
        'mortgage_credit_z',
        'consumer_credit_z',
        'bad_history',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z',
        'self_employed',
        'married'
    ]]

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression using statsmodels Logit (MLE)
    # Wrap in try/except to surface convergence issues
    logit = sm.Logit(y, X)
    try:
        results = logit.fit(disp=False)
    except Exception as e:
        # If Logit fails to converge, fallback to GLM with Binomial family (often more stable)
        results = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # Return the fitted model results object (user can call .summary(), .params, .get_margeff(), etc.)
    return results


