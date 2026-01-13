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
    # Work on a copy
    df = df.copy()

    # Keep only columns we need and drop rows with missing values in those
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried',
                     'religiousness', 'education', 'occupation', 'rating']
    # If some of these columns are missing from input, raise informative error
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    df = df.dropna(subset=required_cols)

    # Create binary indicator for presence of children
    # Accept 'yes'/'no' (case-insensitive) or other encodings; anything starting with 'y' -> 1
    df['ChildrenYes'] = df['children'].apply(lambda x: 1 if str(x).strip().lower().startswith('y') else 0).astype(int)

    # Binary indicator for any affair in the past year
    df['AffairAny'] = (df['affairs'].astype(float) > 0).astype(int)

    # Recode gender to numeric: male = 1, female = 0. If other values present, treat 'male' substring as male.
    df['gender_male'] = df['gender'].apply(lambda x: 1 if str(x).strip().lower().startswith('m') else 0).astype(int)

    # Interaction term for moderation test (Children * Gender)
    df['Children_gender_interaction'] = df['ChildrenYes'] * df['gender_male']

    # Standardize continuous controls (zero mean, unit sd) to aid interpretation and numerical stability
    def z(col: pd.Series) -> pd.Series:
        col = pd.to_numeric(col, errors='coerce')
        m = col.mean()
        s = col.std(ddof=0)
        if s == 0 or np.isnan(s):
            # If no variation, return zeros
            return (col - m).fillna(0.0)
        return ((col - m) / s).fillna(0.0)

    df['age_z'] = z(df['age'])
    df['yearsmarried_z'] = z(df['yearsmarried'])
    df['religiousness_z'] = z(df['religiousness'])
    df['education_z'] = z(df['education'])
    df['occupation_z'] = z(df['occupation'])
    df['rating_z'] = z(df['rating'])

    # Ensure affairs is numeric (it encodes frequency categories). Keep raw coding for count model.
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Final drop of any rows that ended up with NA in model columns
    model_cols = ['affairs', 'AffairAny', 'ChildrenYes', 'gender_male', 'Children_gender_interaction',
                  'age_z', 'yearsmarried_z', 'religiousness_z', 'education_z', 'occupation_z', 'rating_z']
    df = df.dropna(subset=model_cols)

    # Return transformed dataframe with all variables needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # Assumes df has been transformed with transform(df)
    # Prepare design matrix
    model_cols = ['ChildrenYes', 'gender_male', 'Children_gender_interaction',
                  'age_z', 'yearsmarried_z', 'religiousness_z', 'education_z', 'occupation_z', 'rating_z']

    X = df[model_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    results = {}

    # 1) Logistic regression for probability of any affair (binary outcome)
    y_logit = df['AffairAny'].astype(float)
    try:
        logit_mod = sm.Logit(y_logit, X)
        logit_res = logit_mod.fit(disp=False)
        results['logit'] = logit_res
    except Exception as e:
        # If Logit fails (e.g., perfect separation), attempt GLM with binomial family as a fallback
        try:
            glm_binom = sm.GLM(y_logit, X, family=sm.families.Binomial())
            glm_binom_res = glm_binom.fit()
            results['logit_glm_binomial'] = glm_binom_res
        except Exception as e2:
            results['logit_error'] = str(e)

    # 2) Count model for frequency of affairs. Use Negative Binomial GLM to allow overdispersion.
    y_count = df['affairs'].astype(float)
    try:
        nb_mod = sm.GLM(y_count, X, family=sm.families.NegativeBinomial())
        nb_res = nb_mod.fit()
        results['neg_binomial'] = nb_res
    except Exception as e:
        # If negative binomial fails, fall back to Poisson and record the error
        try:
            pois_mod = sm.GLM(y_count, X, family=sm.families.Poisson())
            pois_res = pois_mod.fit()
            results['poisson'] = pois_res
            results['neg_binomial_error'] = str(e)
        except Exception as e2:
            results['neg_binomial_error'] = str(e)

    # Return a dictionary of model fit results objects (or error messages). Callers can inspect summaries via .summary().
    return results


