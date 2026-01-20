from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling the effect of having children on extramarital affairs.
    Produces the following final columns used by the model:
      - 'affairs' (dependent variable, numeric count)
      - 'Children' (binary independent variable: 1=yes, 0=no)
      - 'gender_male' (binary control / moderator: 1=male, 0=female)
      - centered continuous controls: 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c'
      - interaction: 'Children_gender' = Children * gender_male

    The function drops rows with missing values in any of the variables required for the model.
    """
    df = df.copy()

    # Ensure affairs is numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Map children to binary indicator (tolerant to capitalization / different encodings)
    df['Children'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    # If some values are encoded as 1/0 already or boolean, coerce those too
    df.loc[df['Children'].isnull(), 'Children'] = df.loc[df['Children'].isnull(), 'children'].replace({1: 1, 0: 0}).astype(float)

    # Map gender to binary male indicator
    df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    # If other encodings present, attempt common fallbacks (M/F)
    df.loc[df['gender_male'].isnull(), 'gender_male'] = df.loc[df['gender_male'].isnull(), 'gender'].astype(str).str.strip().str.lower().map({'m':1, 'f':0})

    # Convert numeric controls to numeric (coerce errors)
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the dependent variable or main independent variable or moderator or controls
    required = ['affairs', 'Children', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required)

    # Center continuous controls (improves interpretability and numerical stability)
    df['age_c'] = df['age'] - df['age'].mean()
    df['yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()
    df['religiousness_c'] = df['religiousness'] - df['religiousness'].mean()
    df['education_c'] = df['education'] - df['education'].mean()
    df['occupation_c'] = df['occupation'] - df['occupation'].mean()
    df['rating_c'] = df['rating'] - df['rating'].mean()

    # Create interaction term for children * gender
    df['Children_gender'] = df['Children'] * df['gender_male']

    # Final small sanity-check: ensure affairs is integer-like and non-negative
    # (we keep as numeric count; modeling will treat it as count)
    df['affairs'] = df['affairs'].clip(lower=0)

    # Return only columns needed for modeling to keep the dataframe compact (but keep original columns too)
    # Model uses the derived columns; keep original for traceability
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression for the number of affairs as a function of having children,
    controlling for demographic and marital covariates and allowing the effect of children
    to differ by gender (interaction).

    Primary model: Negative Binomial GLM (accounts for overdispersion commonly present in count data).
    We also fit a Poisson model for comparison. Robust (HC0) standard errors are reported.

    Returns a dictionary with fitted models and summaries.
    """
    # select variables for the model
    y = df['affairs']
    X = df[['Children', 'gender_male', 'Children_gender', 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c']]

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial (GLM) with robust standard errors
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit(cov_type='HC0')
    except Exception:
        # Fallback: statsmodels has a discrete NegativeBinomial; if GLM NB fails try discrete model
        try:
            nb_disc = sm.discrete.discrete_model.NegativeBinomial(endog=y, exog=X).fit(disp=False)
            nb_model = nb_disc
        except Exception as e:
            raise RuntimeError(f"Negative binomial model failed: {e}")

    # Fit Poisson for comparison (robust SE)
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type='HC0')

    # Prepare a compact results object: we return the fitted model objects and summaries
    results = {
        'nb_model': nb_model,
        'poisson_model': poisson_model,
        'nb_summary': nb_model.summary() if hasattr(nb_model, 'summary') else None,
        'poisson_summary': poisson_model.summary()
    }

    return results


