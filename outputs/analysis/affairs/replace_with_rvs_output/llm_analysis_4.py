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
    Transform the raw Fair (1978) affairs dataset into a modeling dataframe.

    Outputs (columns required for the model):
      - affair_count: integer count of extramarital affairs (from 'affairs')
      - HasChildren: binary indicator (1 = 'yes', 0 = 'no') from 'children'
      - gender_male: binary indicator (1 = male, 0 = female) from 'gender'
      - age, yearsmarried, religiousness, education, occupation, rating: numeric controls

    Steps:
      - Coerce key columns to numeric where appropriate
      - Map categorical yes/no to 1/0
      - Drop rows with missing values in any variable needed for the model
    """
    df = df.copy()

    # Create affair_count from 'affairs' (coerce non-numeric to NaN first)
    df['affair_count'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Map children (yes/no) to binary indicator HasChildren
    # Be robust to capitalization / whitespace
    df['HasChildren'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Map gender to binary male indicator
    df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

    # Ensure numeric controls are numeric (coerce invalids to NaN)
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in any of the variables used in the model
    required_cols = ['affair_count', 'HasChildren', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_cols)

    # Convert affair_count to integer (counts). If there are non-integer codes they will be truncated.
    # The original coding uses values like 0,1,2,3,7,12; keep them as integers.
    df['affair_count'] = df['affair_count'].astype(int)

    # Make sure HasChildren and gender_male are integer type 0/1
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['gender_male'] = df['gender_male'].astype(int)

    # Return dataframe containing at least the columns used in the model (keeps other columns too)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a count model appropriate for a dependent variable with many zeros and overdispersion.

    Primary approach: Zero-Inflated Negative Binomial (ZINB) with the same covariates in the
    count and inflation equations. The model estimates whether having children is associated
    with a lower expected count of extramarital affairs, controlling for demographic and
    marriage-related covariates.

    Returns the fitted results object (statsmodels result). If ZINB fails to converge, falls
    back to a Negative Binomial GLM.
    """
    import statsmodels.api as sm
    try:
        # Import ZINB implementation
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    except Exception:
        ZeroInflatedNegativeBinomialP = None

    # Prepare exogenous matrix (controls + treatment)
    exog_cols = ['HasChildren', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    X = df[exog_cols].astype(float)
    X = sm.add_constant(X)
    y = df['affair_count'].astype(int)

    # Try ZINB first
    if ZeroInflatedNegativeBinomialP is not None:
        try:
            mod = ZeroInflatedNegativeBinomialP(endog=y, exog=X, exog_infl=X, inflation='logit')
            # suppress optimizer output; increase iterations if necessary
            res = mod.fit(method='newton', maxiter=200, disp=0)
            print(res.summary())
            return res
        except Exception as e:
            print('ZINB failed:', str(e))
            print('Falling back to Negative Binomial GLM...')

    # Fallback: Negative Binomial GLM (does not model zero-inflation explicitly)
    import statsmodels.formula.api as smf
    # Recreate a dataframe for formula-based fitting
    df_model = X.copy()
    df_model['affair_count'] = y.values
    formula = 'affair_count ~ HasChildren + gender_male + age + yearsmarried + religiousness + education + occupation + rating'

    nb_res = smf.glm(formula=formula, data=df_model, family=sm.families.NegativeBinomial()).fit()
    print(nb_res.summary())
    return nb_res


