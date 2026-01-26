from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/shuffle_names_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the Fair (Psychology Today) dataset so that the model can estimate the effect of having children
    on the number/frequency of extramarital affairs.

    Outputs a dataframe containing the exact columns referenced in the conceptual variables:
    - NumAffairs (dependent)
    - HasChildren (independent)
    - Age, EducationYears, Religiousness, YearsMarried, MarriageHappiness, IsFemale, Occupation (controls)

    Notes about the (confusing) input metadata:
    - The provided metadata contains mismatches between column names and descriptions. Based on the textual descriptions
      in the schema, we map: 'education' -> reported extramarital frequency (NumAffairs), 'affairs' -> education level,
      and 'age' -> whether there are children in the marriage (yes/no). We document and implement these mappings below.
    """

    df = df.copy()

    # 1) Create dependent variable NumAffairs from the column 'education' (per provided description)
    #    Ensure numeric and non-negative. Non-numeric entries coerced to NaN.
    if 'education' not in df.columns:
        raise KeyError("Expected column 'education' in input dataframe")
    df['NumAffairs'] = pd.to_numeric(df['education'], errors='coerce')

    # 2) Create independent variable HasChildren from column 'age' which (per metadata) actually encodes yes/no for children
    if 'age' not in df.columns:
        raise KeyError("Expected column 'age' in input dataframe")

    def map_has_children(x):
        # Robust mapping for multiple types / capitalizations
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ['yes', 'y', '1', 'true', 't']:
                return 1
            if s in ['no', 'n', '0', 'false', 'f']:
                return 0
            # fallback - try to parse numeric strings
            try:
                v = float(s)
                return 1 if v != 0 else 0
            except Exception:
                return np.nan
        # numeric types
        try:
            v = float(x)
            # some datasets encode yes=1/no=0
            if v == 1:
                return 1
            if v == 0:
                return 0
            # if value is not 0/1, treat >0 as yes
            return 1 if v > 0 else 0
        except Exception:
            return np.nan

    df['HasChildren'] = df['age'].apply(map_has_children)

    # 3) Controls mapping
    # Age -> from 'rating' (which encodes age categories as numeric representative ages)
    df['Age'] = pd.to_numeric(df['rating'], errors='coerce') if 'rating' in df.columns else np.nan

    # EducationYears -> from 'affairs' column (metadata suggests 'affairs' actually describes education level)
    df['EducationYears'] = pd.to_numeric(df['affairs'], errors='coerce') if 'affairs' in df.columns else np.nan

    # Religiousness
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce') if 'religiousness' in df.columns else np.nan

    # YearsMarried -> from 'yearsmarried'
    df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce') if 'yearsmarried' in df.columns else np.nan

    # MarriageHappiness -> from 'rownames' (metadata: self-rating of marriage 1..5)
    df['MarriageHappiness'] = pd.to_numeric(df['rownames'], errors='coerce') if 'rownames' in df.columns else np.nan

    # IsFemale -> derive from the 'children' column which contains 'female'/'male' in the file's samples
    if 'children' in df.columns:
        def map_female(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, str):
                s = x.strip().lower()
                if s in ['female', 'f', 'woman', 'w']:
                    return 1
                if s in ['male', 'm', 'man']:
                    return 0
                # try numeric
                try:
                    v = float(s)
                    return 1 if v == 1 else 0
                except Exception:
                    return np.nan
            try:
                v = float(x)
                return 1 if v == 1 else 0
            except Exception:
                return np.nan

        df['IsFemale'] = df['children'].apply(map_female)
    else:
        df['IsFemale'] = np.nan

    # Occupation raw code
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce') if 'occupation' in df.columns else np.nan

    # 4) Clean: drop rows missing the key dependent or independent variables
    # We need NumAffairs and HasChildren to answer the research question.
    df_model = df.dropna(subset=['NumAffairs', 'HasChildren']).copy()

    # 5) Further minimal cleaning: cap/validate plausible ranges
    # NumAffairs should be non-negative; drop negative or extremely implausible values
    df_model = df_model[df_model['NumAffairs'] >= 0]

    # For interpretability, ensure integer-valued counts where possible (they may be coded categories like 7/12)
    # Keep the raw numeric coding as provided by the survey; do not coerce categories to other values.

    # 6) Keep only final columns needed for the model (preserve original index)
    final_cols = [
        'NumAffairs', 'HasChildren', 'Age', 'EducationYears', 'Religiousness',
        'YearsMarried', 'MarriageHappiness', 'IsFemale', 'Occupation'
    ]

    # If some columns are missing in df_model (not present in the original), create them as NaN so the downstream
    # model code sees consistent column names.
    for c in final_cols:
        if c not in df_model.columns:
            df_model[c] = np.nan

    # Return only the final columns
    return df_model[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit regression models to estimate the effect of HasChildren on NumAffairs, controlling for covariates.

    We fit:
    1) Negative binomial GLM (primary) because NumAffairs is a non-negative count-like variable and the distribution
       of responses is usually overdispersed.
    2) OLS (secondary / robustness) to see sensitivity of coefficient signs/magnitudes.

    Returns a dict with the fitted result objects for further inspection.
    """

    # Work on a copy
    dfm = df.copy()

    # Ensure we have the necessary columns
    required = ['NumAffairs', 'HasChildren', 'Age', 'EducationYears', 'Religiousness',
                'YearsMarried', 'MarriageHappiness', 'IsFemale', 'Occupation']
    missing = [c for c in required if c not in dfm.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Build the design matrix: include constant
    X = dfm[['HasChildren', 'Age', 'EducationYears', 'Religiousness',
             'YearsMarried', 'MarriageHappiness', 'IsFemale', 'Occupation']].copy()

    # Impute or drop rows with missing covariates: for simplicity drop rows with any missing in X or Y
    data = pd.concat([dfm['NumAffairs'], X], axis=1).dropna()
    y = data['NumAffairs']
    X = data.drop(columns=['NumAffairs'])

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    results = {}

    # 1) Negative Binomial GLM (via statsmodels' GLM with NegativeBinomial family)
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
        results['negbin'] = nb_res
    except Exception as e:
        # If the NegativeBinomial GLM fails (rare), store the exception message
        results['negbin_error'] = str(e)

    # 2) OLS (robust standard errors) as a benchmark
    try:
        ols_model = sm.OLS(y, X)
        ols_res = ols_model.fit(cov_type='HC3')  # robust SEs
        results['ols'] = ols_res
    except Exception as e:
        results['ols_error'] = str(e)

    # 3) Report simple summary statistics for the key variables (for diagnostics)
    try:
        descr = data[['NumAffairs', 'HasChildren']].groupby('HasChildren').agg(['count','mean','std','median'])
        results['group_summary_NumAffairs_by_HasChildren'] = descr
    except Exception:
        results['group_summary_error'] = 'failed to compute group summaries'

    return results


