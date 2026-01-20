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
    Prepare the dataset for modeling the effect of gender on mortgage approval.

    Transformations performed:
    - Make a defensive copy of the DataFrame.
    - Drop rows with missing values in variables required for the model.
    - Ensure binary columns are integer type.
    - Standardize continuous predictors (z-score) to aid interpretation and numerical stability.
    - Create an explicit female x black interaction column (female_black) for clarity (the model formula also includes interaction via female*black).
    - Convert occupation to a categorical dtype (model will include it as a categorical control).

    Final dataframe includes the columns used in the model:
      ['accept', 'female', 'black', 'female_black', 'mortgage_credit_z', 'consumer_credit_z',
       'bad_history', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
       'married', 'self_employed', 'occupation']
    """
    # defensive copy
    df = df.copy()

    # Columns required for the planned model (raw names before standardization)
    required = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit', 'bad_history',
        'PI_ratio', 'loan_to_value', 'housing_expense_ratio', 'married', 'self_employed', 'occupation'
    ]

    # Drop rows missing any required column
    df = df.dropna(subset=required)

    # Ensure binary / indicator columns are integers (0/1)
    for col in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed']:
        # convert to numeric then int (handles floats like 0.0/1.0)
        # Use errors='coerce' then fillna with 0 only as a fallback (dropna above should have removed true missing)
        numeric = pd.to_numeric(df[col], errors='coerce')
        # If there are non-numeric indicators like booleans or strings, attempt to map common ones
        if numeric.isna().any():
            # Try mapping common string representations
            mapped = df[col].map({
                True: 1, False: 0,
                '1': 1, '0': 0,
                'yes': 1, 'no': 0,
                'Yes': 1, 'No': 0,
                'Y': 1, 'N': 0,
                'y': 1, 'n': 0,
                'female': 1, 'male': 0,
                'Female': 1, 'Male': 0,
                'F': 1, 'M': 0
            })
            # Use mapped where not null, otherwise fallback to numeric (which may be NaN)
            numeric = numeric.fillna(mapped)
        # final fillna to avoid casting errors; default to 0 if something unexpected remains
        numeric = numeric.fillna(0).astype(int)
        df[col] = numeric

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for stability.
    cont_cols = ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio']
    for col in cont_cols:
        col_numeric = pd.to_numeric(df[col], errors='coerce')
        mean = col_numeric.mean()
        std = col_numeric.std(ddof=0)
        # If std is zero (unlikely) or NaN, create zeros to avoid division by zero
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (col_numeric - mean) / std

    # Explicit interaction column (helpful for diagnostics and checks)
    df['female_black'] = df['female'] * df['black']

    # Ensure occupation is categorical (model will include it with C(occupation))
    df['occupation'] = df['occupation'].astype('category')

    # Keep only columns that will be used in the modeling step to avoid accidental usage of irrelevant columns
    keep_cols = [
        'accept', 'female', 'black', 'female_black',
        'mortgage_credit_z', 'consumer_credit_z', 'bad_history',
        'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z',
        'married', 'self_employed', 'occupation'
    ]

    df = df[keep_cols].reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binary outcome) estimating the effect of applicant gender on mortgage acceptance,
    controlling for key creditworthiness and demographic characteristics. Also estimates the female x black interaction
    to test whether the gender effect differs for Black applicants.

    Model:
      accept ~ female*black + mortgage_credit_z + consumer_credit_z + bad_history
               + PI_ratio_z + loan_to_value_z + housing_expense_ratio_z
               + married + self_employed + C(occupation)

    Returns:
      The fitted statsmodels result object (Logit). Use result.summary() to inspect coefficients and significance.
    """
    # Make a defensive copy
    df = df.copy()

    # Define formula: female*black includes female, black, and their interaction
    formula = (
        'accept ~ female*black + mortgage_credit_z + consumer_credit_z + bad_history '
        '+ PI_ratio_z + loan_to_value_z + housing_expense_ratio_z '
        '+ married + self_employed + C(occupation)'
    )

    # Fit logistic regression (maximum likelihood). Using smf.logit for interpretability of coefficients.
    model_fit = smf.logit(formula=formula, data=df)
    results = model_fit.fit(disp=False)

    # Return the fitted results object. Caller can do results.summary(), results.params, results.get_margeff(), etc.
    return results