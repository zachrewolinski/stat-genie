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
    Transform raw Fair affairs survey dataframe into the analysis dataframe.

    Produces the following columns used in the model:
      - affairs_count: integer count version of 'affairs'
      - HasChildren: binary indicator (1=yes, 0=no) derived from 'children'
      - IsFemale: binary indicator (1=female, 0=male) derived from 'gender'
      - age_c, yearsmarried_c, religiousness_c, education_c, occupation_c, rating_c: mean-centered numeric controls

    Drops rows with missing values in any of the required variables.
    """
    df = df.copy()

    # Required raw columns
    required = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    # Drop rows missing any required raw inputs
    df = df.dropna(subset=required)

    # Dependent variable: integer count of affairs
    # original coding contains numeric representations (0,1,2,3,7,12); keep as-is
    df['affairs_count'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Independent: HasChildren (map yes/no to 1/0). Handle case differences.
    df['HasChildren'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Control: Gender -> IsFemale (1=female, 0=male)
    df['IsFemale'] = df['gender'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})

    # Ensure numeric controls are numeric
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values after conversions
    needed_after = ['affairs_count', 'HasChildren', 'IsFemale', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=needed_after)

    # Convert affairs_count to integer
    df['affairs_count'] = df['affairs_count'].astype(int)
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['IsFemale'] = df['IsFemale'].astype(int)

    # Mean-center continuous controls to aid interpretation / numerical stability
    df['age_c'] = df['age'] - df['age'].mean()
    df['yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()
    df['religiousness_c'] = df['religiousness'] - df['religiousness'].mean()
    df['education_c'] = df['education'] - df['education'].mean()
    df['occupation_c'] = df['occupation'] - df['occupation'].mean()
    df['rating_c'] = df['rating'] - df['rating'].mean()

    # Final dataframe returned contains all original columns plus the derived ones used in the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model appropriate for excess zeros and overdispersion to estimate the association
    between having children and the count of extramarital affairs, controlling for covariates.

    We use a zero-inflated negative binomial (ZINB) model. ZINB models two processes:
      (1) a logit model for structural zeros (always-zero process), and
      (2) a negative binomial count model for the non-structural process.

    Returns the fitted results object.
    """
    # Import here to avoid top-level import issues in some environments
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    import statsmodels.api as sm

    # Endogenous (count) variable
    endog = df['affairs_count'].astype(int)

    # Exogenous regressors for the count model (include constant)
    exog_cols = ['HasChildren', 'IsFemale', 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c']
    exog = sm.add_constant(df[exog_cols], has_constant='add')

    # Exogenous regressors for the inflation (zero) model. We include gender and religiosity
    # as plausible predictors of structural zeros (i.e., never engaging in affairs).
    exog_infl = sm.add_constant(df[['IsFemale', 'religiousness_c']], has_constant='add')

    # Instantiate and fit the Zero-Inflated Negative Binomial model
    model_mod = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')

    # Fit the model. Set disp=0 to suppress verbose output; increase maxiter if convergence warnings appear.
    results = model_mod.fit(disp=0, maxiter=100)

    # Return the fitted results object. Callers can inspect results.summary(), results.params, etc.
    return results


