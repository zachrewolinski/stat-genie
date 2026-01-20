from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/noperturb_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for modeling the effect of having children on reported extramarital affairs.

    Transformations performed:
    - Make a copy of the dataframe to avoid side effects.
    - Drop rows with missing values in key variables used in the analysis.
    - Create a binary Children column: 1 if children == 'yes' (case-insensitive), 0 if 'no'.
    - Create a binary gender_Female column: 1 if gender == 'female', 0 if 'male'.
    - Ensure numeric columns are numeric (coerce when necessary).

    Final dataframe contains at least the following columns used in modeling:
    'affairs', 'Children', 'gender_Female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating'
    """
    df = df.copy()

    # Columns required for analysis
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Create binary Children indicator (1 if 'yes', 0 if 'no') - handle capitalization
    df['Children'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # If mapping produced NaN for unexpected values, coerce to 0 (conservative) but keep note: these rows were not dropped above
    df['Children'] = df['Children'].fillna(0).astype(int)

    # Create binary gender indicator female = 1, male = 0
    df['gender_Female'] = df['gender'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})
    # If other/unexpected entries appear, coerce to 0
    df['gender_Female'] = df['gender_Female'].fillna(0).astype(int)

    # Ensure numeric columns are numeric
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # After coercion, drop rows with missing affairs or key numeric controls (we need valid outcome and controls)
    df = df.dropna(subset=['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count models to estimate the association between having children and extramarital affairs.

    Modeling approach:
    - Main model: Negative binomial Generalized Linear Model (to account for overdispersion in the count outcome).
    - Robustness: Ordinary Least Squares (OLS) regression reported for comparison/interpretability.

    The formula controls for gender, age, years married, religiousness, education, occupation, and marital rating.

    Returns a dictionary with keys 'nb_model' and 'ols_model' containing fitted model result objects.
    """
    import statsmodels.formula.api as smf

    # Ensure we work on a copy
    df = df.copy()

    # Define formula using exact column names produced in transform
    formula = 'affairs ~ Children + gender_Female + age + yearsmarried + religiousness + education + occupation + rating'

    # Fit Negative Binomial (GLM) as main model
    try:
        nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If glm NegativeBinomial fails, raise an informative error
        raise RuntimeError(f'NegativeBinomial GLM failed: {e}')

    # Fit OLS as a robustness check
    ols_model = smf.ols(formula=formula, data=df).fit()

    # Return both fitted result objects for inspection
    results = {
        'nb_model': nb_model,
        'ols_model': ols_model
    }

    return results


