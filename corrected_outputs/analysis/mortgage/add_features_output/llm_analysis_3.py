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
    Transform the raw dataframe into a clean analysis dataframe.

    - Keep only columns necessary for the analysis.
    - Drop rows with missing accept or female values (must have DV and IV).
    - Ensure binary indicators are integer 0/1.
    - Create standardized (z-scored) versions of continuous controls to aid interpretation/stability.
    - Return a dataframe containing the exact columns used in the model.
    """
    df = df.copy()

    # Columns we will use (original names from the dataset schema)
    keep_cols = [
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
        'housing_expense_ratio',
        'denied_PMI'
    ]

    # Keep only these columns if present
    present_cols = [c for c in keep_cols if c in df.columns]

    # Require that the essential columns exist
    if 'accept' not in present_cols or 'female' not in present_cols:
        raise KeyError("Input dataframe must contain 'accept' and 'female' columns")

    df = df[present_cols].copy()

    # Drop rows missing the outcome or primary independent variable
    df = df.dropna(subset=['accept', 'female']).reset_index(drop=True)

    # Ensure binary indicators are integer 0/1 where appropriate
    binary_cols = [c for c in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI'] if c in df.columns]
    for c in binary_cols:
        # Coerce to numeric, turn non-numeric -> NaN, then fill NaN with 0 and cast to int.
        # Finally clamp to 0/1 to be safe.
        series_num = pd.to_numeric(df[c], errors='coerce').fillna(0)
        # Round any floats that are close to integers, then clip to 0/1
        series_int = series_num.round().astype(int).clip(lower=0, upper=1)
        df[c] = series_int

    # Create z-scored continuous controls to put them on comparable scales
    cont_cols = [c for c in ['mortgage_credit', 'consumer_credit', 'PI_ratio', 'loan_to_value', 'housing_expense_ratio'] if c in df.columns]
    for c in cont_cols:
        # coerce to numeric; keep NaNs for observations where conversion fails
        series_num = pd.to_numeric(df[c], errors='coerce')
        mean = series_num.mean()
        std = series_num.std(ddof=0)
        if pd.isna(std) or std == 0:
            # create a column of zeros (float) for stability
            df[c + '_z'] = 0.0
        else:
            df[c + '_z'] = (series_num - mean) / std

    # Final columns to return (must match the conceptual variables exactly)
    final_cols = [
        'accept',
        'female',
        'black',
        'bad_history',
        'married',
        'self_employed',
        'denied_PMI'
    ]
    # add the standardized continuous columns that were created
    for c in ['mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z']:
        if c in df.columns:
            final_cols.append(c)

    # Keep only final columns that exist in the transformed df
    final_cols = [c for c in final_cols if c in df.columns]

    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression of mortgage approval (accept) on female status and controls.

    Returns a dict with:
    - 'model_result': the fitted statsmodels LogitResults object (or GLMResults fallback)
    - 'avg_marginal_effect_female': the average marginal effect of being female on the probability of acceptance (computed as the average difference in predicted probabilities when female=1 vs female=0, holding other covariates at their observed values)

    Note: the function expects the dataframe returned by transform(), containing the columns named in the conceptual variables.
    """
    df = df.copy()

    # Ensure required columns present
    required_cols = ['accept', 'female']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Model input dataframe must contain column '{col}'")

    # Define outcome and predictors (must match transform output column names)
    y = df['accept']

    predictors = [
        'female',
        'black',
        'bad_history',
        'married',
        'self_employed',
        'denied_PMI'
    ]
    # Add standardized continuous controls if present
    for c in ['mortgage_credit_z', 'consumer_credit_z', 'PI_ratio_z', 'loan_to_value_z', 'housing_expense_ratio_z']:
        if c in df.columns:
            predictors.append(c)

    # Keep only predictors that exist in df
    predictors = [p for p in predictors if p in df.columns]

    X = df[predictors]

    # Drop rows with missing data in y or X (required for statsmodels)
    combined = pd.concat([y, X], axis=1)
    combined = combined.dropna()
    if combined.shape[0] == 0:
        raise ValueError("No observations remain after dropping rows with missing values in outcome or predictors.")
    y = combined['accept']
    X = combined[predictors]

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    # Use try/except to surface a helpful error if optimization fails
    try:
        logit_mod = sm.Logit(y, X)
        result = logit_mod.fit(disp=False)
    except Exception:
        # If Logit fails to converge or errors, try GLM with binomial family as fallback
        glm_mod = sm.GLM(y, X, family=sm.families.Binomial())
        result = glm_mod.fit()

    # Compute average marginal effect of female by computing predicted probabilities
    # for each observation with female set to 1 vs 0, holding other covariates equal to their observed values,
    # then averaging the difference.
    X_female1 = X.copy()
    X_female0 = X.copy()
    if 'female' not in X_female1.columns:
        raise KeyError("'female' column not found in predictors")
    X_female1['female'] = 1
    X_female0['female'] = 0

    # Use the fitted model to predict probabilities. result.predict expects the same exog shape as was used in fit.
    p1 = result.predict(X_female1)
    p0 = result.predict(X_female0)
    avg_marginal_effect = (p1 - p0).mean()

    # Return results and the A.M.E. for female
    return {
        'model_result': result,
        'avg_marginal_effect_female': avg_marginal_effect
    }