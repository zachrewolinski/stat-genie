from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw mortgage dataset into a dataframe with the variables used for modeling.

    Produces these final columns (at minimum):
      - is_denied: binary DV (1 denied, 0 accepted) from 'mortgage_credit'
      - is_female: binary IV (1 female, 0 male) from 'consumer_credit'
      - is_black: control from 'bad_history'
      - is_married: control from 'PI_ratio'
      - self_employed: control from 'self_employed'
      - loan_to_value: numeric control from 'loan_to_value'
      - denied_PMI: numeric control from 'denied_PMI'
      - consumer_score: numeric control from 'accept'
      - housing_exp_*: dummy variables derived from 'housing_expense_ratio' (drop_first=True)

    The function checks required columns, coerces types, fills reasonable missing values for continuous controls,
    and returns a dataframe containing only the final columns used in the model.
    """
    df = df.copy()

    # Required source columns (raise clear error if missing)
    required = [
        'mortgage_credit',  # DV: 1 if denied, 0 if accepted
        'consumer_credit',  # IV: 1 if female, 0 if male
        'bad_history',
        'PI_ratio',
        'self_employed',
        'loan_to_value',
        'housing_expense_ratio',
        'denied_PMI',
        'accept'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing DV or IV
    df = df.dropna(subset=['mortgage_credit', 'consumer_credit'])

    # Create dependent and independent variables
    df['is_denied'] = pd.to_numeric(df['mortgage_credit'], errors='coerce').astype(float).round().astype(int)
    df['is_female'] = pd.to_numeric(df['consumer_credit'], errors='coerce').fillna(0).astype(int)

    # Controls: coerce to numeric and fill plausible defaults where appropriate
    df['is_black'] = pd.to_numeric(df['bad_history'], errors='coerce').fillna(0).astype(int)
    df['is_married'] = pd.to_numeric(df['PI_ratio'], errors='coerce').fillna(0).astype(int)
    df['self_employed'] = pd.to_numeric(df['self_employed'], errors='coerce').fillna(0).astype(int)
    df['loan_to_value'] = pd.to_numeric(df['loan_to_value'], errors='coerce')
    df['denied_PMI'] = pd.to_numeric(df['denied_PMI'], errors='coerce')
    df['consumer_score'] = pd.to_numeric(df['accept'], errors='coerce')

    # For continuous controls, fill missing with median (conservative imputation)
    for cont in ['loan_to_value', 'denied_PMI', 'consumer_score']:
        if df[cont].isnull().any():
            df[cont] = df[cont].fillna(df[cont].median())

    # Housing expense ratio: treat as categorical and create dummies. Keep levels > 1 as dummies (drop_first to avoid multicollinearity)
    # Coerce to integer categories where possible
    df['housing_expense_ratio'] = pd.to_numeric(df['housing_expense_ratio'], errors='coerce')
    # If housing_expense_ratio is largely integer-like but floats with NaNs, create dummies from rounded values
    df['housing_expense_ratio'] = df['housing_expense_ratio'].round().fillna(0).astype(int)
    dummies = pd.get_dummies(df['housing_expense_ratio'], prefix='housing_exp', drop_first=True)
    # Ensure consistent dummy columns for levels 2,3,4 if present; create any missing to keep column set stable
    for lvl in [2, 3, 4]:
        col = f'housing_exp_{lvl}'
        if col not in dummies.columns:
            dummies[col] = 0
    # Keep only the expected dummy columns in sorted order
    dummy_cols = [f'housing_exp_{lvl}' for lvl in [2, 3, 4] if f'housing_exp_{lvl}' in dummies.columns]
    dummies = dummies[dummy_cols]

    # Concatenate dummies
    df = pd.concat([df, dummies], axis=1)

    # Final column ordering
    final_cols = [
        'is_denied',
        'is_female',
        'is_black',
        'is_married',
        'self_employed',
        'loan_to_value',
        'denied_PMI',
        'consumer_score'
    ] + dummy_cols

    # Return only the final columns (safe subset for modeling)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting mortgage denial from applicant gender (is_female)
    while adjusting for controls. Returns the fitted statsmodels results object.

    Model form:
      logit(P(is_denied=1)) = beta0 + beta1*is_female + Beta_controls * controls

    Controls included: is_black, is_married, self_employed, loan_to_value, denied_PMI, consumer_score,
    and housing expense dummies (housing_exp_2, housing_exp_3, housing_exp_4) if present.

    The function will attempt sm.Logit; if there is perfect separation or solver issues, it will fall back to
    sm.GLM with Binomial family.
    """
    df = df.copy()

    # Confirm required model columns exist
    if 'is_denied' not in df.columns:
        raise KeyError("Transformed dataframe must contain 'is_denied' column as DV")

    # Define predictors
    base_controls = [
        'is_female',
        'is_black',
        'is_married',
        'self_employed',
        'loan_to_value',
        'denied_PMI',
        'consumer_score'
    ]
    # include any housing dummies present in df
    housing_dummies = [c for c in df.columns if c.startswith('housing_exp_')]
    X_cols = base_controls + housing_dummies

    # Ensure predictors exist in dataframe
    missing = [c for c in X_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing predictor columns in transformed df: {missing}")

    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['is_denied'].astype(int)

    # Fit logistic regression (maximum likelihood) with robust fallback
    try:
        logit_model = sm.Logit(y, X)
        res = logit_model.fit(disp=False)
    except Exception as e:
        # Fall back to GLM(Binomial) which is often more numerically stable
        glm_model = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm_model.fit()

    # Return the fitted results object. Caller can call res.summary() to inspect coefficients and stats.
    return res


