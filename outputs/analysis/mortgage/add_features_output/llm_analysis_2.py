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
    Transform the raw Boston Fed mortgage dataset into the final dataframe used for modeling.

    Steps:
    - Make a local copy
    - Select the variables needed for the analysis
    - Drop rows with missing values on those variables
    - Ensure binary variables are integer 0/1
    - Standardize continuous covariates into z-scores (new columns with _z suffix)
    - Create the outcome column 'accepted' from the original 'accept' column

    Returns a dataframe containing exactly the columns used by the statistical model.
    """
    df = df.copy()

    # Columns required for the analysis (raw names as present in the dataset)
    required_cols = [
        'accept',            # original outcome (1 accepted, 0 denied)
        'female',
        'black',
        'mortgage_credit',
        'consumer_credit',
        'PI_ratio',
        'loan_to_value',
        'bad_history',
        'married',
        'self_employed',
        'housing_expense_ratio',
        'denied_PMI'
    ]

    # Keep only required columns if they exist
    existing_required = [c for c in required_cols if c in df.columns]
    df = df[existing_required]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=existing_required)

    # Coerce binary indicators to ints (if they are floats)
    for bin_col in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']:
        if bin_col in df.columns:
            df[bin_col] = df[bin_col].astype(int)

    # Create final dependent variable column 'accepted' (1 accepted, 0 denied)
    df['accepted'] = df['accept'].astype(int)

    # Continuous variables to standardize (create new _z columns)
    cont_vars = [c for c in ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio'] if c in df.columns]

    # If any continuous variable is non-numeric, coerce to numeric
    for c in cont_vars:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Re-drop rows that became NA after coercion
    df = df.dropna(subset=cont_vars + ['accepted', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI'])

    # Compute z-scores (mean 0, sd 1) for continuous controls
    for c in cont_vars:
        sd = df[c].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            # If zero variance, produce a zero column to avoid division by zero
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (df[c] - df[c].mean()) / sd

    # Final columns used in the model (exact names referenced in the model function)
    final_columns = [
        'accepted',
        'female',
        'black',
        'bad_history',
        'married',
        'self_employed',
        'denied_PMI',
        # standardized continuous controls
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z'
    ]

    # Keep only the columns that exist (in case some original dataset variants lack some columns)
    final_existing = [c for c in final_columns if c in df.columns]
    df_final = df[final_existing].reset_index(drop=True)

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression to estimate the effect of applicant gender on acceptance probability,
    controlling for observable applicant and loan characteristics.

    Model: accepted ~ female + controls
    Uses statsmodels Logit (maximum likelihood logistic regression).

    Returns the fitted results object (statsmodels.discrete.discrete_model.BinaryResults).
    Also prints a brief summary.
    """
    df = df.copy()

    # Dependent variable
    y = df['accepted']

    # Predictor/controls: exact column names must match those produced by transform()
    X_cols = [
        'female',
        'black',
        'bad_history',
        'married',
        'self_employed',
        'denied_PMI',
        'mortgage_credit_z',
        'consumer_credit_z',
        'PI_ratio_z',
        'loan_to_value_z',
        'housing_expense_ratio_z'
    ]

    # Keep only columns that exist in the incoming dataframe
    X_cols = [c for c in X_cols if c in df.columns]

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression. Use try/except to provide a helpful error if fitting fails.
    try:
        model = sm.Logit(y, X)
        results = model.fit(disp=False)
    except Exception as e:
        # If Logit fails (perfect separation, etc.), fall back to GLM with binomial family
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        results = glm.fit()

    # Print summary for quick inspection
    print(results.summary())

    # Return the results object for downstream use (coefs, conf_int, predict, etc.)
    return results


