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
    """
    Prepare the Fair (1978) affairs dataset for modeling.
    Produces the following columns required by the models:
      - affairs: numeric count (original column coerced to numeric)
      - AnyAffair: binary indicator (1 if affairs > 0)
      - HasChildren: binary indicator derived from 'children' (1=yes, 0=no)
      - gender_male: binary indicator derived from 'gender' (1=male, 0=female), fallback to 'gender_mf' if needed
      - age, yearsmarried, religiousness, education, rating, occupation preserved as numeric controls

    Drops rows with missing values in any of the modeling columns.
    """

    df = df.copy()

    # Ensure affairs is numeric (coerce non-numeric to NaN)
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Map children to binary HasChildren (1=yes, 0=no). Accepts 'yes'/'no' strings.
    # If children is already numeric coded, try to coerce numeric values as well.
    df['children_str'] = df['children'].astype(str).str.strip().str.lower()
    df['HasChildren'] = df['children_str'].map({'yes': 1, 'no': 0})
    # If mapping failed but original is numeric-like (0/1), try coercion
    try:
        # attempt to coerce values like 0/1 stored as numbers or strings
        children_num = pd.to_numeric(df['children'], errors='coerce')
        df.loc[df['HasChildren'].isna() & children_num.notna(), 'HasChildren'] = children_num
    except Exception:
        pass

    # Create binary gender indicator gender_male from 'gender' if available; fallback to 'gender_mf' numeric column
    df['gender_str'] = df['gender'].astype(str).str.strip().str.lower()
    df['gender_male'] = df['gender_str'].map({'male': 1, 'female': 0})
    if 'gender_mf' in df.columns:
        # fill missing gender_male by numeric gender_mf (if present and plausible)
        df['gender_male'] = df['gender_male'].fillna(df['gender_mf'])

    # Binary indicator for whether any affair occurred
    df['AnyAffair'] = (df['affairs'] > 0).astype(int)

    # Ensure occupation is numeric
    df['occupation'] = pd.to_numeric(df['occupation'], errors='coerce')

    # Convert core controls to numeric where appropriate
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop temporary helper columns
    df = df.drop(columns=[c for c in ['children_str', 'gender_str'] if c in df.columns])

    # Final list of columns required by the models
    required_cols = ['affairs', 'AnyAffair', 'HasChildren', 'age', 'yearsmarried', 'religiousness', 'education', 'rating', 'gender_male', 'occupation']

    # Drop rows with missing values in any of the required columns
    df = df.dropna(subset=required_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to assess whether having children is associated
    with engagement in extramarital affairs.

    1) Logistic regression (binary outcome AnyAffair): estimates the association
       between HasChildren and the probability of any affair, controlling for covariates.
    2) Negative binomial GLM (count outcome 'affairs'): estimates association with
       the count of affairs (including zeros) while allowing for overdispersion.

    Returns a dict with keys 'logit' and 'negbin' whose values are fitted statsmodels results objects.
    """

    df = df.copy()

    # Specify covariates (same covariates used in both models)
    exog_vars = ['HasChildren', 'age', 'yearsmarried', 'religiousness', 'education', 'rating', 'gender_male', 'occupation']

    # Prepare design matrix
    X = df[exog_vars]
    X = sm.add_constant(X, has_constant='add')

    results = {}

    # 1) Logistic regression for probability of any affair
    y_logit = df['AnyAffair']
    try:
        logit_model = sm.Logit(y_logit, X)
        logit_res = logit_model.fit(disp=False)
        results['logit'] = logit_res
    except Exception as e:
        # Fall back to a regularized fit if perfect separation or convergence problems
        try:
            logit_res = sm.Logit(y_logit, X).fit_regularized(method='l1', disp=False)
            results['logit'] = logit_res
        except Exception as e2:
            results['logit_error'] = str(e2)

    # 2) Negative binomial GLM for count of affairs (models counts including zeros)
    y_count = df['affairs']
    try:
        negbin_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial())
        negbin_res = negbin_model.fit()
        results['negbin'] = negbin_res
    except Exception as e:
        # If NegativeBinomial fails, try Poisson as fallback
        try:
            poisson_res = sm.GLM(y_count, X, family=sm.families.Poisson()).fit()
            results['negbin_fallback_poisson'] = poisson_res
        except Exception as e2:
            results['negbin_error'] = str(e2)

    return results


