from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried',
                     'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing values in variables needed for the model
    df = df.dropna(subset=required_cols)

    # Dependent variable: use the original 'affairs' numeric coding as a count-like outcome
    # Convert to integer (affairs already coded as numeric in the dataset description).
    df['affair_count'] = pd.to_numeric(df['affairs'], errors='coerce').astype(float)

    # If any non-finite or negative values appear, drop them
    df = df[df['affair_count'].notna()]
    df = df[df['affair_count'] >= 0]

    # Independent variable: children -> binary indicator
    # Map common textual encodings to 0/1; if already binary/category, map accordingly
    df['children_binary'] = df['children'].astype(str).str.strip().str.lower().map({
        'yes': 1,
        'y': 1,
        '1': 1,
        'true': 1,
        'no': 0,
        'n': 0,
        '0': 0,
        'false': 0
    })
    # If mapping produced NaNs (unexpected categories), attempt to interpret numeric
    if df['children_binary'].isna().any():
        # try numeric coercion
        coerced = pd.to_numeric(df.loc[df['children_binary'].isna(), 'children'], errors='coerce')
        df.loc[df['children_binary'].isna(), 'children_binary'] = coerced.fillna(0).clip(0,1)

    df['children_binary'] = df['children_binary'].astype(int)

    # Control: gender -> binary male indicator (1=male, 0=female)
    df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map({
        'male': 1,
        'm': 1,
        'man': 1,
        'female': 0,
        'f': 0,
        'woman': 0
    })
    # If gender mapping produced NaNs, try numeric coercion fallback
    if df['gender_male'].isna().any():
        coerced_g = pd.to_numeric(df.loc[df['gender_male'].isna(), 'gender'], errors='coerce')
        df.loc[df['gender_male'].isna(), 'gender_male'] = coerced_g.fillna(0).clip(0,1)
    df['gender_male'] = df['gender_male'].astype(int)

    # Interaction term to test moderation of children effect by gender
    df['children_gender_interaction'] = df['children_binary'] * df['gender_male']

    # Ensure control variables are numeric
    numeric_controls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in numeric_controls:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['affair_count', 'children_binary', 'gender_male'] + numeric_controls)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    # Build design matrix for Negative Binomial regression
    # Columns used in the model must match those created in transform()
    exog_cols = [
        'children_binary',
        'gender_male',
        'children_gender_interaction',
        'age',
        'yearsmarried',
        'religiousness',
        'education',
        'occupation',
        'rating'
    ]

    # Safety check
    missing = [c for c in exog_cols + ['affair_count'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Prepare exogenous matrix and add constant
    exog = df[exog_cols].astype(float)
    exog = sm.add_constant(exog, has_constant='add')
    endog = df['affair_count'].astype(float)

    # Fit Negative Binomial via GLM (robust for overdispersion relative to Poisson)
    # Use default log link. If convergence or other issues arise, consider sm.NegativeBinomial from discrete models.
    try:
        nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial())
        results = nb_model.fit()
    except Exception:
        # Fallback: fit Poisson if NB fails
        poisson_model = sm.GLM(endog, exog, family=sm.families.Poisson())
        results = poisson_model.fit()

    # Print summary for quick inspection (caller can examine returned results object)
    print(results.summary())

    return results


