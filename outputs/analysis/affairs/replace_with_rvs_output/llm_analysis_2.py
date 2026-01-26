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

    Steps performed:
    - Work on a copy of df to avoid side effects.
    - Map children and gender to binary indicators: Children (1/0) and Female (1/0).
    - Create interaction term Children_Female for testing moderation.
    - Standardize continuous covariates (z-scores) used in the model for stability and interpretability.
    - Drop rows with missing values in any variables used by the model.

    Returns dataframe containing the columns used in the statistical model (and keeps original affairs column).
    """
    df = df.copy()

    # Ensure expected columns exist
    expected = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in input dataframe: {missing}")

    # Map children to binary indicator (handle 'yes'/'no' or numeric factor)
    def map_children(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl in ['yes', 'y', '1', 'true', 't']:
                return 1
            if xl in ['no', 'n', '0', 'false', 'f']:
                return 0
        # otherwise try numeric coercion
        try:
            xv = float(x)
            # If dataset uses factor coding like 1/2, attempt to map: assume 1=yes or use >0
            # Here we assume values >0 indicate yes; but preserve 1/0 mapping if present
            if xv == 1:
                return 1
            if xv == 0:
                return 0
        except Exception:
            pass
        return np.nan

    df['Children'] = df['children'].apply(map_children)

    # Map gender to Female indicator (1 = female, 0 = male). Robust to various text cases.
    def map_female(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl.startswith('f') or 'female' in xl:
                return 1
            if xl.startswith('m') or 'male' in xl:
                return 0
        # if numeric, try heuristic: assume 0/1 where 1 is female? But safer to coerce to NaN
        try:
            xv = float(x)
            # if only two numeric codes present, user should inspect; here treat nan for numeric unexpected values
            return np.nan
        except Exception:
            return np.nan

    df['Female'] = df['gender'].apply(map_female)

    # Create interaction term for moderation test
    df['Children_Female'] = df['Children'] * df['Female']

    # Standardize continuous controls (z-scores). Use ddof=0 to compute population std for consistent scaling.
    cont_vars = ['age', 'yearsmarried', 'education', 'religiousness', 'occupation', 'rating']
    for v in cont_vars:
        # coerce to numeric
        df[v] = pd.to_numeric(df[v], errors='coerce')
        mean = df[v].mean()
        std = df[v].std(ddof=0)
        if std == 0 or pd.isna(std):
            # avoid division by zero; set to 0
            df[v + '_z'] = (df[v] - mean)
        else:
            df[v + '_z'] = (df[v] - mean) / std

    # Ensure affairs is numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Select final columns required by the model and drop rows with missing values in these columns
    final_cols = [
        'affairs',
        'Children',
        'Female',
        'Children_Female',
        'age_z',
        'yearsmarried_z',
        'education_z',
        'religiousness_z',
        'occupation_z',
        'rating_z'
    ]

    df_final = df[final_cols].dropna()

    # (Optional) cast integer-like indicators to int for readability
    df_final['Children'] = df_final['Children'].astype(int)
    df_final['Female'] = df_final['Female'].astype(int)
    df_final['Children_Female'] = (df_final['Children'] * df_final['Female']).astype(int)

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit count-regression models testing whether having children decreases engagement in extramarital affairs.

    Modeling strategy:
    - Use Negative Binomial (NB) GLM to handle overdispersion relative to Poisson.
    - Fit a main-effects model with Children and controls.
    - Fit a second model adding the Children x Female interaction to test whether the effect of children differs by gender.

    Returns a dictionary with both fitted statsmodels results objects.
    """
    # Ensure the required columns are present
    required = ['affairs', 'Children', 'Female', 'Children_Female', 'age_z', 'yearsmarried_z', 'education_z', 'religiousness_z', 'occupation_z', 'rating_z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Response
    y = df['affairs']

    # Main-effects predictors
    X_main = df[['Children', 'Female', 'age_z', 'yearsmarried_z', 'education_z', 'religiousness_z', 'occupation_z', 'rating_z']]
    X_main = sm.add_constant(X_main)

    # Fit Negative Binomial GLM (handles overdispersion)
    # Note: statsmodels' GLM with NegativeBinomial family estimates an extra parameter for overdispersion.
    model_nb_main = sm.GLM(y, X_main, family=sm.families.NegativeBinomial()).fit()

    # Interaction model: add Children_Female
    X_int = df[['Children', 'Female', 'Children_Female', 'age_z', 'yearsmarried_z', 'education_z', 'religiousness_z', 'occupation_z', 'rating_z']]
    X_int = sm.add_constant(X_int)
    model_nb_int = sm.GLM(y, X_int, family=sm.families.NegativeBinomial()).fit()

    # Return both fitted model results for inspection
    return {
        'nb_main': model_nb_main,
        'nb_with_interaction': model_nb_int
    }


