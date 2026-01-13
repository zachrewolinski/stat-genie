from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/noperturb_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw fair dataset into a cleaned dataframe ready for count-modeling.

    Output dataframe columns used in the model:
      - 'affairs' (dependent variable, numeric)
      - 'HasChildren' (0/1 main IV)
      - 'Female' (0/1 control)
      - 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating' (controls)

    Steps:
      - Standardize/cast types
      - Create binary indicators
      - Drop rows with missing values in variables needed for the model
    """

    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure affairs is numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Normalize children column to binary HasChildren (1 if yes, 0 if no)
    # Handle varied capitalization and possible missing values
    df['children'] = df['children'].astype(str).str.strip().str.lower()
    df['HasChildren'] = df['children'].map({'yes': 1, 'y': 1, 'no': 0, 'n': 0})

    # If mapping produced NaN but original entries look like '1'/'0', attempt numeric fallback
    mask_missing_children = df['HasChildren'].isna()
    if mask_missing_children.any():
        # try to coerce numeric
        coerced = pd.to_numeric(df.loc[mask_missing_children, 'children'], errors='coerce')
        df.loc[mask_missing_children, 'HasChildren'] = coerced.fillna(df.loc[mask_missing_children, 'HasChildren'])

    # Create Female indicator from gender (1 female, 0 male)
    df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    df['Female'] = df['gender'].apply(lambda x: 1 if str(x).startswith('f') else (0 if str(x).startswith('m') else np.nan))

    # Ensure control variables numeric
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only rows with non-missing values for dependent, IV, and controls
    required_cols = ['affairs', 'HasChildren', 'Female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_cols)

    # Cast HasChildren and Female to integers
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['Female'] = df['Female'].astype(int)

    # Ensure affairs is integer (the dataset encodes groups like 7 or 12 to represent ranges; keep as given)
    df['affairs'] = df['affairs'].astype(int)

    # Final dataframe includes the columns used by the model
    final_cols = ['affairs', 'HasChildren', 'Female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> object:
    """
    Fit a zero-inflated negative binomial (ZINB) model predicting number of affairs
    using HasChildren as the main independent variable and adjusting for controls.

    If ZINB fails to converge, fall back to a Negative Binomial GLM.

    Returns the fitted results object (statsmodels.results).
    """

    # Import the required model class
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    except Exception:
        ZeroInflatedNegativeBinomialP = None

    # Prepare endogenous and exogenous matrices
    endog = df['affairs'].astype(float)
    exog_vars = ['HasChildren', 'Female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    exog = sm.add_constant(df[exog_vars], has_constant='add')

    # For the inflation (zero) equation, use the same set of controls (including HasChildren)
    exog_infl = exog.copy()

    results = None
    if ZeroInflatedNegativeBinomialP is not None:
        try:
            zinb_mod = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')
            # Fit with reasonable iteration limits; suppress convergence output
            results = zinb_mod.fit(disp=False, maxiter=200)
            return results
        except Exception as e:
            # If ZINB fails (convergence or other numerical issues), fall back
            print('ZINB failed with error:', e)

    # Fallback: Negative Binomial GLM (counts but no zero-inflation)
    try:
        nb_mod = sm.GLM(endog, exog, family=sm.families.NegativeBinomial())
        results = nb_mod.fit()
        return results
    except Exception as e:
        # Final fallback: Poisson (if NB also fails)
        print('Negative Binomial failed with error:', e)
        poisson_mod = sm.GLM(endog, exog, family=sm.families.Poisson())
        results = poisson_mod.fit()
        return results


