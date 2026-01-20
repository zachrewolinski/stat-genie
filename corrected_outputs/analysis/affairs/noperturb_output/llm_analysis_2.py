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
    # Work on a copy
    df = df.copy()

    # Ensure relevant columns exist
    required_cols = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]

    # If any required columns missing, raise an informative error
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # Strip whitespace and coerce textual columns to lowercase for robust mapping
    df['children_str'] = df['children'].astype(str).str.strip().str.lower()
    df['gender_str'] = df['gender'].astype(str).str.strip().str.lower()

    # Map children to binary: common values 'yes'/'no'. Attempt robust matching (y/n prefixes).
    df['children_binary'] = np.where(
        df['children_str'].str.startswith('y'), 1,
        np.where(df['children_str'].str.startswith('n'), 0, np.nan)
    )

    # Map gender to male indicator (1=male, 0=female). If other encodings exist, attempt to infer by prefix.
    df['gender_male'] = np.where(
        df['gender_str'].str.startswith('m'), 1,
        np.where(df['gender_str'].str.startswith('f'), 0, np.nan)
    )

    # Coerce numeric variables to numeric dtype (if they are strings)
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in any of the variables that will be used in models
    model_cols = ['affairs', 'children_binary', 'gender_male', 'age', 'yearsmarried',
                  'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=model_cols)

    # Ensure affairs is a non-negative integer count
    # Round or coerce if necessary (data already uses integer-coded counts). Keep as int.
    df['affairs'] = df['affairs'].astype(int)
    df['children_binary'] = df['children_binary'].astype(int)
    df['gender_male'] = df['gender_male'].astype(int)

    # Final dataframe contains both original variables and created binaries used in modeling
    # Return only columns necessary for modeling (but keep originals that may be helpful)
    final_cols = [
        'affairs', 'children_binary', 'gender_male', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits a primary count model appropriate for overdispersed count data with excess zeros (Zero-Inflated Negative Binomial)
    and an OLS model as a simple robustness check. Returns fitted results objects.

    Expects the transformed dataframe to contain the columns:
      - 'affairs' (dependent count)
      - 'children_binary' (primary independent variable)
      - 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating' (controls)
    """
    # Imports local to the modeling function
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Prepare exogenous regressors for the count model (include intercept)
    exog_vars = ['children_binary', 'gender_male', 'age', 'yearsmarried',
                 'religiousness', 'education', 'occupation', 'rating']
    exog = sm.add_constant(df[exog_vars], has_constant='add')

    # For the inflation (zero) part, use a subset of predictors that plausibly explain structural zeros
    infl_vars = ['gender_male', 'age', 'yearsmarried', 'religiousness']
    exog_infl = sm.add_constant(df[infl_vars], has_constant='add')

    # Fit Zero-Inflated Negative Binomial (ZINB) model
    zinb_mod = ZeroInflatedNegativeBinomialP(endog=df['affairs'], exog=exog, exog_infl=exog_infl, inflation='logit', p=1)
    try:
        zinb_res = zinb_mod.fit(method='bfgs', maxiter=200, disp=False)
    except Exception:
        # fallback to default optimizer if bfgs fails
        zinb_res = zinb_mod.fit(maxiter=200, disp=False)

    # Fit an OLS for a simple robustness check (note: OLS is not ideal for counts but is informative)
    ols_mod = sm.OLS(df['affairs'], exog)
    ols_res = ols_mod.fit()

    # Return both results so the researcher can inspect ZINB coefficients and OLS robustness
    return {
        'zinb_results': zinb_res,
        'ols_results': ols_res
    }


