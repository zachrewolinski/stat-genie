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
    Transform the raw Fair affairs dataframe into the analysis dataframe.

    Produces the following columns used in the model:
      - AffairsCount: numeric copy of 'affairs'
      - ChildrenBinary: 1 if children == 'yes', 0 if 'no'
      - GenderMale: 1 if gender == 'male', 0 if 'female'
      - Age, Yearsmarried, Religiousness, Education, Occupation, Rating: numeric copies
      - Children_GenderInteraction: ChildrenBinary * GenderMale
      - const: constant column for model intercept

    Drops rows with missing values in any of the variables used for modeling.
    """
    # Copy / normalize outcome
    df = df.copy()

    # Ensure affairs numeric
    df['AffairsCount'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Binary children variable: 'yes' -> 1, 'no' -> 0
    df['ChildrenBinary'] = df['children'].map(lambda x: 1 if str(x).strip().lower() == 'yes' else (0 if str(x).strip().lower() == 'no' else np.nan))

    # Gender binary: male = 1, female = 0
    df['GenderMale'] = df['gender'].map(lambda x: 1 if str(x).strip().lower() == 'male' else (0 if str(x).strip().lower() == 'female' else np.nan))

    # Numeric controls: ensure numeric dtype and coerce errors to NaN
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    df['Yearsmarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Interaction term: children * gender
    df['Children_GenderInteraction'] = df['ChildrenBinary'] * df['GenderMale']

    # Add constant column for modeling
    df['const'] = 1.0

    # Keep only rows with complete data on variables we will use
    required_cols = [
        'AffairsCount', 'ChildrenBinary', 'GenderMale', 'Age', 'Yearsmarried',
        'Religiousness', 'Education', 'Occupation', 'Rating', 'Children_GenderInteraction', 'const'
    ]

    df = df.dropna(subset=required_cols)

    # Optional: ensure AffairsCount is non-negative integer-like; keep as-is because coding includes 0,1,2,3,7,12
    # Cast AffairsCount to integer if it is whole-valued
    if np.all(np.mod(df['AffairsCount'].dropna(), 1) == 0):
        df['AffairsCount'] = df['AffairsCount'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a zero-inflated negative binomial model predicting AffairsCount from ChildrenBinary
    controlling for gender, age, years married, religiousness, education, occupation, and marriage rating.

    The model allows a separate zero-inflation (logit) submodel that includes ChildrenBinary and GenderMale
    (i.e., zeros may be systematically related to having children and gender).

    Returns the fitted results object (statsmodels wrapper). Use results.summary() to inspect coefficients.
    """
    # Import the ZINB class
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Select regressors for the count model (exog) and for the inflation (zero) model (exog_infl).
    # exog contains the controls and interaction.
    exog_cols = ['const', 'ChildrenBinary', 'GenderMale', 'Children_GenderInteraction',
                 'Age', 'Yearsmarried', 'Religiousness', 'Education', 'Occupation', 'Rating']
    exog_infl_cols = ['const', 'ChildrenBinary', 'GenderMale']

    exog = df[exog_cols]
    exog_infl = df[exog_infl_cols]
    endog = df['AffairsCount']

    # Fit the Zero-Inflated Negative Binomial model. Use logit inflation and allow dispersion to be estimated.
    # Use method 'newton' for robust convergence and limit output.
    zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')
    try:
        results = zinb.fit(method='newton', maxiter=100, disp=0)
    except Exception:
        # fallback to default fit if newton fails
        results = zinb.fit(disp=0)

    return results


