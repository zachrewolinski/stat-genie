from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (Psychology Today) dataset into a clean dataframe for modeling the effect
    of having children on engagement in extramarital affairs.

    Final dataframe columns used in models:
      - affairs: numeric (original 'affairs' column)
      - AnyAffair: binary indicator (1 if affairs > 0, else 0)
      - LogAffairs: log(affairs + 1)
      - HasChildren: binary indicator (1 if respondent reports children in the marriage, 0 otherwise) -- constructed from column 'age' which holds the 'children in marriage' factor in this schema
      - SexFemale: binary (1 if 'children' column == 'female', 0 if 'male') -- in this schema 'children' column actually contains gender
      - Age: numeric (from 'rating' column which codes age groups)
      - Yearsmarried: numeric (from 'yearsmarried')
      - education, religiousness, occupation, rownames preserved as controls

    The transform is robust to common messy encodings in the provided schema.
    """
    df = df.copy()

    # Standardize numeric columns (coerce non-numeric to NaN)
    numeric_cols = ['affairs', 'education', 'rating', 'yearsmarried', 'religiousness', 'occupation', 'rownames']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 1) Create HasChildren from the column 'age' according to provided schema ('age' = factor: are there children?)
    #    Accept common encodings: 'yes'/'no', 'Y'/'N', 1/0, True/False
    def to_binary_yes(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ['yes', 'y', 'true', 't', '1']:
            return 1
        if s in ['no', 'n', 'false', 'f', '0']:
            return 0
        # if it's numeric and not 0 treat as 1
        try:
            num = float(x)
            if num == 0:
                return 0
            else:
                return 1
        except Exception:
            return np.nan

    if 'age' in df.columns:
        df['HasChildren'] = df['age'].apply(to_binary_yes)
    else:
        df['HasChildren'] = np.nan

    # 2) Create SexFemale from column 'children' which (in this schema) holds gender labels
    if 'children' in df.columns:
        def sex_from_children(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip().lower()
            if s in ['female', 'f', 'woman', 'w']:
                return 1
            if s in ['male', 'm', 'man']:
                return 0
            # fallback: try numeric
            try:
                num = float(x)
                # if coded 1/2 (unknown), we can't safely map; return NaN
                return np.nan
            except Exception:
                return np.nan
        df['SexFemale'] = df['children'].apply(sex_from_children)
    else:
        df['SexFemale'] = np.nan

    # 3) Create AnyAffair and LogAffairs
    if 'affairs' in df.columns:
        df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
        df['AnyAffair'] = (df['affairs'] > 0).astype(float)  # keep float to be compatible with statsmodels
        df['LogAffairs'] = np.log(df['affairs'].fillna(0) + 1)
    else:
        df['affairs'] = np.nan
        df['AnyAffair'] = np.nan
        df['LogAffairs'] = np.nan

    # 4) Pull other controls into cleaner column names expected by the model
    # Age: use 'rating' (this dataset encodes age groups in 'rating')
    if 'rating' in df.columns:
        df['Age'] = pd.to_numeric(df['rating'], errors='coerce')
    else:
        df['Age'] = np.nan

    # Ensure yearsmarried is numeric and present
    if 'yearsmarried' in df.columns:
        df['Yearsmarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    else:
        # fallback: some schemas put years married in 'gender' column; attempt that if present
        if 'gender' in df.columns:
            df['Yearsmarried'] = pd.to_numeric(df['gender'], errors='coerce')
        else:
            df['Yearsmarried'] = np.nan

    # Keep education, religiousness, occupation, rownames as provided (already coerced above)
    # If column names are missing, create them as NaN so downstream code doesn't break
    for col in ['education', 'religiousness', 'occupation', 'rownames']:
        if col not in df.columns:
            df[col] = np.nan

    # 5) Final selection and dropping rows with missing key variables
    final_cols = ['affairs', 'AnyAffair', 'LogAffairs', 'HasChildren', 'SexFemale', 'Age', 'Yearsmarried', 'education', 'religiousness', 'occupation', 'rownames']
    # Keep only rows with non-missing HasChildren and AnyAffair and the primary controls (we'll drop rows missing HasChildren or AnyAffair)
    df = df.copy()
    df = df[final_cols]

    # Drop rows missing HasChildren or AnyAffair
    df = df.dropna(subset=['HasChildren', 'AnyAffair'])

    # For convenience, also coerce remaining columns to numeric where appropriate (they already are),
    # but avoid dropping rows for every control to preserve sample size; the model will drop rows with any NaNs.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run two complementary models to estimate the association between having children and
    engagement in extramarital affairs:
      1) Logistic regression predicting AnyAffair (binary) -- primary test of whether having children
         decreases the probability of having had at least one affair.
      2) OLS regression predicting LogAffairs = log(affairs + 1) -- tests whether having children is
         associated with (log) intensity of affairs.

    Returns a dict with keys 'logit' and 'ols' containing fitted statsmodels result objects.
    """
    import statsmodels.api as sm

    df = df.copy()

    # Define independent / control variables used in both models
    model_vars = ['HasChildren', 'SexFemale', 'Age', 'Yearsmarried', 'education', 'religiousness', 'rownames', 'occupation']

    # Drop rows with missing values in any of the predictors or the relevant DVs for each model separately
    # Logistic model dataset
    df_logit = df.dropna(subset=['AnyAffair'] + model_vars)
    X_logit = df_logit[model_vars]
    X_logit = sm.add_constant(X_logit, has_constant='add')
    y_logit = df_logit['AnyAffair']

    results = {}

    # Fit logistic regression (use Logit); if it fails due to separation or convergence, capture the exception
    try:
        logit_model = sm.Logit(y_logit, X_logit)
        logit_res = logit_model.fit(disp=False)
        results['logit'] = logit_res
    except Exception as e:
        # Fall back to GLM with binomial family which is more stable in some cases
        try:
            glm_binom = sm.GLM(y_logit, X_logit, family=sm.families.Binomial())
            glm_res = glm_binom.fit()
            results['logit'] = glm_res
            results['logit_fallback'] = True
        except Exception as e2:
            results['logit_error'] = str(e)

    # OLS model for intensity (log of affairs + 1)
    df_ols = df.dropna(subset=['LogAffairs'] + model_vars)
    X_ols = df_ols[model_vars]
    X_ols = sm.add_constant(X_ols, has_constant='add')
    y_ols = df_ols['LogAffairs']

    try:
        ols_model = sm.OLS(y_ols, X_ols)
        ols_res = ols_model.fit()
        results['ols'] = ols_res
    except Exception as e:
        results['ols_error'] = str(e)

    # Return results dict (statsmodels result objects are returned for further inspection)
    return results


