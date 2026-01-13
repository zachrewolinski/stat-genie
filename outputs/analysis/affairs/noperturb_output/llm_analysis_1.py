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
    Produces the following columns used in modeling:
      - Affairs: numeric copy of original 'affairs' column (count, 0..12)
      - AnyAffair: binary indicator (1 if Affairs>0, else 0)
      - HasChildren: binary indicator derived from 'children' (1 if 'yes', 0 if 'no')
      - Gender_Male: binary indicator derived from 'gender' (1 if 'male', 0 if 'female')
      - Age, YearsMarried, Religiousness, Education, Occupation, Rating: cleaned numeric controls

    The function drops rows with missing values in variables required for the models.
    """

    # Work on a copy
    df = df.copy()

    # Standardize column names that we will use (preserve original columns but create new standardized ones)
    # 1) Affairs (count) - ensure numeric
    if 'affairs' in df.columns:
        df['Affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
    else:
        df['Affairs'] = np.nan

    # 2) AnyAffair - binary indicator for whether any affair occurred
    df['AnyAffair'] = (df['Affairs'] > 0).astype(int)

    # 3) HasChildren: map 'children' factor ('yes'/'no') to 1/0. Accept capitalization variants.
    if 'children' in df.columns:
        df['HasChildren'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    else:
        df['HasChildren'] = np.nan

    # 4) Gender_Male: map 'gender' ('male'/'female') to 1/0
    if 'gender' in df.columns:
        df['Gender_Male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    else:
        df['Gender_Male'] = np.nan

    # 5) Numeric control variables: Age, YearsMarried, Religiousness, Education, Occupation, Rating
    # Use existing column names but create standardized names
    df['Age'] = pd.to_numeric(df['age'], errors='coerce') if 'age' in df.columns else np.nan
    df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce') if 'yearsmarried' in df.columns else np.nan
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce') if 'religiousness' in df.columns else np.nan
    df['Education'] = pd.to_numeric(df['education'], errors='coerce') if 'education' in df.columns else np.nan
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce') if 'occupation' in df.columns else np.nan
    df['Rating'] = pd.to_numeric(df['rating'], errors='coerce') if 'rating' in df.columns else np.nan

    # 6) Drop rows with missing values in variables necessary for modeling
    required_cols = [
        'Affairs', 'AnyAffair', 'HasChildren', 'Gender_Male',
        'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'Rating'
    ]
    df = df.dropna(subset=required_cols)

    # 7) Optional: convert integer types
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['Gender_Male'] = df['Gender_Male'].astype(int)
    df['AnyAffair'] = df['AnyAffair'].astype(int)
    df['Affairs'] = df['Affairs'].astype(float)

    # 8) (Optional) create a small summary column indicating whether counts are large; not used in primary models but useful for diagnostics
    df['Affairs_gt0'] = (df['Affairs'] > 0).astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to assess the effect of children on extramarital affairs:
      1) Logistic regression for probability of any affair (AnyAffair)
      2) Negative binomial regression for the count of affairs among respondents (Affairs)

    Returns a dict with keys 'logit_result' and 'nb_result' containing fitted statsmodels results objects.
    """

    results = {}

    # Prepare the design matrix for controls and the IV
    controls = ['Gender_Male', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'Rating']
    iv = ['HasChildren']
    model_vars = iv + controls

    # Ensure required columns exist
    for col in model_vars + ['AnyAffair', 'Affairs']:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe passed to model().")

    # Add constant
    X = sm.add_constant(df[model_vars])

    # 1) Logistic regression for AnyAffair
    y_logit = df['AnyAffair'].astype(float)
    try:
        logit_mod = sm.Logit(y_logit, X)
        logit_res = logit_mod.fit(disp=False)
    except Exception as e:
        # Fall back to GLM binomial if Logit has convergence issues
        logit_mod = sm.GLM(y_logit, X, family=sm.families.Binomial())
        logit_res = logit_mod.fit()

    results['logit_result'] = logit_res

    # 2) Negative binomial regression for count data (Affairs)
    # Use the whole sample but a count model naturally handles zeros; alternatively, one could model positives only.
    y_nb = df['Affairs'].astype(float)
    try:
        nb_mod = sm.GLM(y_nb, X, family=sm.families.NegativeBinomial())
        nb_res = nb_mod.fit()
    except Exception as e:
        # If NegativeBinomial family is not stable, fall back to Poisson with robust cov
        nb_mod = sm.GLM(y_nb, X, family=sm.families.Poisson())
        nb_res = nb_mod.fit(cov_type='HC0')

    results['nb_result'] = nb_res

    # Return both fitted model results for inspection
    return results


