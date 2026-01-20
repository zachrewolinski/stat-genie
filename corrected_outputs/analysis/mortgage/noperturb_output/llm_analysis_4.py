from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw HMDA-style dataset to a modeling-ready dataframe.

    Steps performed:
    - Drop an optional index column 'Unnamed: 0' if present.
    - Create a binary dependent variable 'approved' using 'accept' when available; otherwise use 1 - deny.
    - Ensure key columns are numeric and drop rows with missing values in variables required for the model.
    - Create an interaction term 'female_black' to test whether race moderates the effect of gender.

    The returned dataframe contains at minimum the columns referenced in the conceptual model:
    ['female', 'approved', 'black', 'female_black', 'housing_expense_ratio', 'self_employed',
     'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value',
     'denied_PMI']
    """
    df = df.copy()

    # Drop extraneous unnamed index column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Derive the dependent variable 'approved'
    if 'accept' in df.columns:
        df['approved'] = pd.to_numeric(df['accept'], errors='coerce')
    elif 'deny' in df.columns:
        # if only deny is available, approved = 1 - deny
        df['approved'] = 1 - pd.to_numeric(df['deny'], errors='coerce')
    else:
        raise ValueError("Dataset must contain either 'accept' or 'deny' column to derive approval outcome.")

    # Ensure binary gender indicator exists and is numeric
    if 'female' not in df.columns:
        raise ValueError("Dataset must contain 'female' column as the primary independent variable.")
    df['female'] = pd.to_numeric(df['female'], errors='coerce')

    # List all columns required for the model
    required_cols = [
        'female',
        'approved',
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

    # Coerce columns that exist in the dataframe to numeric types (to handle strings / mixed types)
    for c in required_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the core variables used in modeling
    df = df.dropna(subset=['female', 'approved'])

    # If any additional required control columns are present, drop rows missing them
    # (we require all listed controls to be present for the main model)
    controls_present = [c for c in required_cols if c not in ['female', 'approved'] and c in df.columns]
    if len(controls_present) > 0:
        df = df.dropna(subset=controls_present)

    # Create interaction term for testing moderation (female x black)
    if 'black' in df.columns:
        df['female_black'] = df['female'] * df['black']
    else:
        # create zero column if race not present to keep model code consistent (but user should typically have 'black')
        df['black'] = 0
        df['female_black'] = 0

    # Final safety: ensure binary columns are 0/1 integers where appropriate
    for bin_col in ['female', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI', 'approved']:
        if bin_col in df.columns:
            # round and clip to 0/1 to handle small float artifacts
            df[bin_col] = df[bin_col].round().clip(0, 1).astype(int)

    # Return only columns needed for modeling (keeps dataframe compact and explicit)
    model_cols = [
        'female', 'approved', 'black', 'female_black', 'housing_expense_ratio', 'self_employed',
        'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value',
        'denied_PMI'
    ]
    present_model_cols = [c for c in model_cols if c in df.columns]
    return df[present_model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to estimate the effect of gender on mortgage approval,
    controlling for applicant financial and demographic characteristics.

    Model specification (primary):
      logit(P(approved=1)) = beta0 + beta1 * female + beta2 * black + beta3 * (female*black)
                              + beta4 * housing_expense_ratio + beta5 * self_employed + ...

    The function returns the fitted statsmodels LogitResults object for the full model (includes the female x black interaction).
    """
    import statsmodels.api as sm  # local import to avoid top-level dependency issues

    # Define regressors used in the model; require them to be present in df
    X_cols = [
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

    missing = [c for c in X_cols + ['approved'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"The following required columns are missing from the transformed dataframe: {missing}")

    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['approved'].astype(int)

    # Fit logistic regression (maximum likelihood). Use robust standard errors as an option.
    logit_model = sm.Logit(y, X)
    try:
        results = logit_model.fit(disp=False)
    except Exception:
        # fallback: use a GLM binomial if Logit has convergence issues
        results = sm.GLM(y, X, family=sm.families.Binomial()).fit()

    # It's useful to also fit a simpler model without the interaction to inspect main effect of female
    # (returned as part of results_dict for convenience)
    X_no_inter = X.drop(columns=['female_black'])
    try:
        results_no_inter = sm.Logit(y, X_no_inter).fit(disp=False)
    except Exception:
        results_no_inter = sm.GLM(y, X_no_inter, family=sm.families.Binomial()).fit()

    # Return both fitted models so the analyst can compare the main effect and the moderated effect
    return {
        'model_with_interaction': results,
        'model_without_interaction': results_no_inter
    }