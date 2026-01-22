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
    Transform the raw Fair (1978) affairs dataset into a modeling dataframe.

    Produces the following columns (exact names used by the model):
      - affairs (dependent variable, numeric)
      - Children (binary independent variable: 1 if children present, 0 if not)
      - Gender_Male (binary control: 1 = male, 0 = female)
      - age, yearsmarried, religiousness, education, occupation, rating (numeric controls)

    Steps:
      - drop rows with missing values in required fields
      - coerce types, map categorical text values to numeric
      - return dataframe with only the columns needed for modeling
    """
    df = df.copy()

    # Ensure required columns exist
    required = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing values in key variables
    df = df.dropna(subset=required)

    # Encode Children: original values are 'yes'/'no' (factor). Map to 1/0.
    df['Children'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    # If there are other encodings (e.g., 'Yes', 'No', 1, 0), handle them
    df['Children'] = df['Children'].fillna(df['children'].replace({1: 1, 0: 0}))

    # Drop rows where Children mapping failed
    df = df.dropna(subset=['Children'])
    df['Children'] = df['Children'].astype(int)

    # Encode gender: map 'male' -> 1, 'female' -> 0
    df['Gender_Male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    # If already coded differently, attempt to coerce
    df['Gender_Male'] = df['Gender_Male'].fillna(df['gender'].replace({'m': 1, 'f': 0}))
    df = df.dropna(subset=['Gender_Male'])
    df['Gender_Male'] = df['Gender_Male'].astype(int)

    # Ensure numeric columns are numeric; coerce errors to NaN then drop
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols)

    # Keep only the exact columns needed for modeling (and in the exact names expected)
    out_cols = ['affairs', 'Children', 'Gender_Male', 'age', 'yearsmarried',
                'religiousness', 'education', 'occupation', 'rating']
    df_out = df[out_cols].reset_index(drop=True)

    return df_out


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to estimate the association between having children and
    extramarital affairs, controlling for covariates.

    Models:
      1) Robust OLS (linear) as a baseline (heteroskedasticity-robust SEs)
      2) Zero-Inflated Negative Binomial (ZINB) to account for count-like outcome with many zeros

    Returns a dictionary containing both fitted results objects:
      { 'ols_results': <statsmodels.regression.linear_model.RegressionResultsWrapper>,
        'zinb_results': <statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialResults> }

    Note: the calling environment can print .summary() on each result object.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Ensure input contains the required columns (names set by transform)
    req = ['affairs', 'Children', 'Gender_Male', 'age', 'yearsmarried',
           'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in req if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # 1) OLS with robust SEs
    formula = 'affairs ~ Children + Gender_Male + age + yearsmarried + religiousness + education + occupation + rating'
    ols_res = smf.ols(formula, data=df).fit(cov_type='HC3')

    # 2) Zero-Inflated Negative Binomial
    # Prepare exogenous matrices. Use a richer set for the count model and a smaller set for inflation.
    exog_vars = ['Children', 'Gender_Male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    exog = sm.add_constant(df[exog_vars], has_constant='add')

    # For the inflation (zero) model use a subset that plausibly predicts structural zeros (e.g., Children, Gender, age, yearsmarried)
    exog_infl = sm.add_constant(df[['Children', 'Gender_Male', 'age', 'yearsmarried']], has_constant='add')

    endog = df['affairs'].astype(int)

    # Fit ZINB. Set disp=False to avoid printing optimizer output.
    zinb_model = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')
    try:
        zinb_res = zinb_model.fit(disp=False, maxiter=1000)
    except Exception:
        # In case of convergence issues, try a few different optimizers / increased iterations
        zinb_res = zinb_model.fit(disp=False, method='bfgs', maxiter=2000)

    return {
        'ols_results': ols_res,
        'zinb_results': zinb_res
    }


