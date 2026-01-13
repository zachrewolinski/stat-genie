from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Fair (affairs) dataset for modeling.
    Returns DataFrame containing the columns used in the statistical model:
      - affairs (dependent variable, numeric)
      - children_yes (0/1)
      - gender_male (0/1)
      - children_gender (interaction)
      - age_c, yearsmarried_c, education_c, rating_c (centered continuous covariates)
      - religiousness, occupation (controls)
    """
    df = df.copy()

    # Keep only rows with non-missing dependent and main independent variable
    required = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required)

    # Ensure numeric columns are numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['yearsmarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['education'] = pd.to_numeric(df['education'], errors='coerce')
    df['occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows that became NA after coercion
    df = df.dropna(subset=['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating'])

    # Recode children to binary indicator: 1 if 'yes', 0 if 'no'. Be robust to case and whitespace.
    df['children'] = df['children'].astype(str).str.strip().str.lower()
    df['children_yes'] = df['children'].map(lambda x: 1 if x == 'yes' else 0)

    # Recode gender: 1 if male, 0 if female. Be robust to capitalization.
    df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    df['gender_male'] = df['gender'].map(lambda x: 1 if x in ('male', 'm') else 0)

    # Create interaction term for moderation tests
    df['children_gender'] = df['children_yes'] * df['gender_male']

    # Center continuous covariates to improve interpretability and reduce collinearity with interactions
    df['age_c'] = df['age'] - df['age'].mean()
    df['yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()
    df['education_c'] = df['education'] - df['education'].mean()
    df['rating_c'] = df['rating'] - df['rating'].mean()

    # Keep only columns needed for modeling (but retain others if desired)
    cols_needed = ['affairs', 'children_yes', 'gender_male', 'children_gender', 'age_c', 'yearsmarried_c', 'religiousness', 'education_c', 'occupation', 'rating_c']
    df = df[cols_needed].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression appropriate for the affairs outcome.

    Rationale for model choice:
    - The dependent variable 'affairs' is a count-like variable with many zeros
      and a limited top code (values like 0,1,2,3,7,12). A zero-inflated count
      model accommodates excess zeros while modeling the count process.
    - We estimate a Zero-Inflated Poisson (ZIP) with the same covariates in the
      inflation (zero) model. The main parameter of interest is the coefficient
      on children_yes and the children_yes x gender_male interaction.

    Returns the fitted model result object (statsmodels results instance).
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    # Check that required columns are present
    required = ['affairs', 'children_yes', 'gender_male', 'children_gender', 'age_c', 'yearsmarried_c', 'religiousness', 'education_c', 'occupation', 'rating_c']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe required for model: {missing}")

    # Endogenous variable
    endog = df['affairs'].astype(float)

    # Exogenous regressors for count model: include main effect of children, gender,
    # their interaction, and controls
    exog_vars = ['children_yes', 'gender_male', 'children_gender', 'age_c', 'yearsmarried_c', 'religiousness', 'education_c', 'occupation', 'rating_c']
    exog = sm.add_constant(df[exog_vars], has_constant='add')

    # For the inflation (zero) model we use the same covariates (alternative specifications are possible)
    exog_infl = sm.add_constant(df[exog_vars], has_constant='add')

    # Fit Zero-Inflated Poisson
    # Note: if convergence or overdispersion is an issue, consider ZeroInflatedNegativeBinomialP
    zip_model = ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl, inflation='logit')

    try:
        result = zip_model.fit(method='bfgs', maxiter=200, disp=False)
    except Exception:
        # fallback to default fit settings if BFGS fails
        result = zip_model.fit(disp=False)

    # Return the fitted results object for downstream inspection (coefficients, p-values, etc.)
    return result


