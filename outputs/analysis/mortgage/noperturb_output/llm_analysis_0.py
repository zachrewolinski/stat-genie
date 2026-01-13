from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/noperturb_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a cleaned dataframe ready for modeling.

    Steps performed:
    - Make a shallow copy of the input dataframe.
    - Require presence of the main columns ('accept', 'female').
    - Drop rows with missing values for the dependent variable ('accept') and the main independent variable ('female').
    - Convert relevant columns to numeric where appropriate.
    - Impute missing numeric control values with the column median.
    - Impute missing binary control values with the column mode (or 0 if mode cannot be determined).
    - Ensure binary indicators are integers (0/1).
    - Create an interaction column 'female_black' = female * black to test moderation (intersection of gender and race).
    - Drop rows that still have missing values in any of the model columns.

    Returns the cleaned dataframe containing at minimum the columns referenced in the conceptual model.
    """
    df = df.copy()

    # Ensure required columns exist
    required = [
        'accept',  # dependent variable
        'female'   # independent variable
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input dataframe")

    # Drop rows missing the DV or IV
    df = df.dropna(subset=required)

    # List of control columns expected
    control_cols = [
        'black',
        'housing_expense_ratio',
        'self_employed',
        'married',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'denied_PMI'
    ]

    # Convert known columns to numeric where present
    for col in control_cols + ['female', 'accept']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Numeric columns to impute with median
    numeric_impute = ['housing_expense_ratio', 'mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value']
    for col in numeric_impute:
        if col in df.columns:
            median_val = df[col].median()
            if np.isnan(median_val):
                # if the column is entirely missing or non-numeric, fill with 0
                median_val = 0
            df[col] = df[col].fillna(median_val)

    # Binary / categorical columns to impute with mode
    binary_cols = ['black', 'self_employed', 'married', 'bad_history', 'denied_PMI']
    for col in binary_cols:
        if col in df.columns:
            if df[col].isnull().all():
                df[col] = df[col].fillna(0).astype(int)
            else:
                modes = df[col].mode()
                fill = int(modes.iloc[0]) if not modes.empty else 0
                df[col] = df[col].fillna(fill).astype(int)

    # Ensure accept and female are integer 0/1
    df['female'] = df['female'].astype(int)
    df['accept'] = df['accept'].astype(int)

    # Create interaction term for moderation test: female * black
    if 'black' in df.columns:
        df['female_black'] = (df['female'] * df['black']).astype(int)
    else:
        # if 'black' missing, create column of zeros to keep modeling pipeline stable
        df['female_black'] = 0

    # Final model columns (only include control columns that exist in the dataframe)
    model_cols = ['accept', 'female', 'black', 'female_black']
    model_cols += [c for c in control_cols if c in df.columns]

    # Drop any rows with missing values in the final model columns
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to estimate the effect of applicant gender on mortgage acceptance,
    controlling for creditworthiness and other covariates and testing moderation by race (black).

    Model specification (primary):
      accept ~ female + black + female_black + housing_expense_ratio + self_employed + married
               + mortgage_credit + consumer_credit + bad_history + PI_ratio + loan_to_value + denied_PMI

    The function returns the fitted statsmodels results object. If Logit fails to converge or raises
    an exception due to e.g. perfect separation, it falls back to a GLM with a binomial family.
    """
    # Use the transform function to ensure we have a cleaned dataframe (safe if a transformed df is passed)
    if not {'female_black'}.issubset(df.columns):
        df = transform(df)
    else:
        # still run general cleaning for any missingness
        df = transform(df)

    # Define outcome and predictors
    y = df['accept']

    # Build list of predictors — include only those columns that exist after transform
    predictor_candidates = [
        'female',
        'black',
        'female_black',
        'housing_expense_ratio',
        'self_employed',
        'married',
        'mortgage_credit',
        'consumer_credit',
        'bad_history',
        'PI_ratio',
        'loan_to_value',
        'denied_PMI'
    ]
    X_cols = [c for c in predictor_candidates if c in df.columns]

    if len(X_cols) == 0:
        raise ValueError('No predictor columns available for modeling')

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood) and fall back to GLM if necessary
    try:
        logit_model = sm.Logit(y, X)
        results = logit_model.fit(disp=False)
    except Exception as e:
        # Often issues arise due to perfect separation or convergence; try GLM (robust fallback)
        glm_model = sm.GLM(y, X, family=sm.families.Binomial())
        results = glm_model.fit()

    return results


