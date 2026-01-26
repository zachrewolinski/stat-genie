from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/positive_leading_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Fair (1978) affairs dataset for modeling.

    Produces the following columns used in the model:
      - affairs: dependent variable (numeric count/frequency)
      - Children: binary indicator (1 if children in marriage, 0 otherwise)
      - IsFemale: binary gender indicator (1 female, 0 male)
      - age_c: age mean-centered
      - yearsmarried_c: years married mean-centered
      - religiousness, education, occupation, rating: numeric controls

    Drops rows with missing values in any of the above variables.
    """
    df = df.copy()

    # Basic required columns
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried',
                     'religiousness', 'education', 'occupation', 'rating']

    # Ensure required columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows where the key outcome or the key IV is missing
    df = df.dropna(subset=['affairs', 'children'])

    # Map children to binary (1=yes, 0=no). Accepts variants in capitalization.
    df['Children'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Map gender to binary female indicator (1=female, 0=male). If other levels, will become NaN and dropped.
    df['IsFemale'] = df['gender'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})

    # Ensure numeric columns are numeric; coerce errors to NaN so they get dropped below
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with any missing values in the variables we will use
    use_cols = ['affairs', 'Children', 'IsFemale', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=use_cols)

    # Mean-center continuous predictors to help interpretation
    df['age_c'] = df['age'] - df['age'].mean()
    df['yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()

    # Keep only the columns needed for modeling (and return original 'affairs' as DV)
    out_cols = ['affairs', 'Children', 'IsFemale', 'age_c', 'yearsmarried_c', 'religiousness', 'education', 'occupation', 'rating']
    return df[out_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a Negative Binomial GLM to estimate the effect of having children on number/frequency
    of extramarital affairs, controlling for demographic and marriage-related covariates.

    Returns a dictionary with:
      - 'nb': fitted Negative Binomial GLM results (statsmodels results instance)
      - 'nb_robust': same NB results but with robust (HC3) covariance matrix applied
      - 'poisson': Poisson GLM fitted results (for sensitivity / comparison)

    Interpretation: The coefficient on 'Children' is the log multiplicative effect on expected
    affair counts. A negative coefficient supports the hypothesis that having children decreases
    engagement in extramarital affairs.
    """
    import statsmodels.api as sm

    # Make a defensive copy
    df = df.copy()

    # Ensure the DV and IV exist in dataframe
    required = ['affairs', 'Children', 'IsFemale', 'age_c', 'yearsmarried_c', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Define outcome and predictors
    y = df['affairs']
    X = df[['Children', 'IsFemale', 'age_c', 'yearsmarried_c', 'religiousness', 'education', 'occupation', 'rating']]
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial GLM (handles overdispersion better than Poisson)
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    nb_res = nb_model.fit()

    # Robust covariance (HC3) for inference
    try:
        nb_res_robust = nb_res.get_robustcov_results(cov_type='HC3')
    except Exception:
        # fallback: if robust results not available, return the original
        nb_res_robust = nb_res

    # Fit Poisson for comparison / sensitivity
    pois_model = sm.GLM(y, X, family=sm.families.Poisson())
    pois_res = pois_model.fit()

    # Print a short summary (optional) and return results
    print("Negative Binomial coefficient for Children:", nb_res.params.get('Children'))
    print(nb_res.summary())

    return {
        'nb': nb_res,
        'nb_robust': nb_res_robust,
        'poisson': pois_res
    }


