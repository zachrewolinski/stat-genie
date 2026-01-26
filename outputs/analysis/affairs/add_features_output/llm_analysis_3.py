from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (1978) affairs dataset to create the variables used in modeling.

    Output columns required by the model:
      - affairs: original count variable (kept as-is)
      - AnyAffair: binary indicator (1 if affairs > 0, else 0)
      - Children: binary indicator (1 if children == 'yes', 0 if 'no')
      - GenderMale: binary indicator (1 if gender == 'male', 0 if 'female')
      - age_z, yearsmarried_z, religiousness_z, education_z, occupation_z, rating_z: standardized numeric controls
      - Children_Gender: interaction term (Children * GenderMale)

    The function drops rows with missing values in columns required for the analysis.
    """
    df = df.copy()

    # Keep the original affairs column
    # Drop rows missing any variables needed for our analysis
    required_cols = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]
    df = df.dropna(subset=required_cols)

    # Binary Children: dataset uses 'yes'/'no'
    df['Children'] = df['children'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    # If mapping produced any NaN (unexpected labels), drop those rows
    df = df[df['Children'].notna()]
    df['Children'] = df['Children'].astype(int)

    # Gender binary: create GenderMale (1=male, 0=female)
    df['GenderMale'] = df['gender'].astype(str).str.lower().map({'male': 1, 'female': 0})
    df = df[df['GenderMale'].notna()]
    df['GenderMale'] = df['GenderMale'].astype(int)

    # Binary AnyAffair indicator
    df['AnyAffair'] = (df['affairs'].astype(float) > 0).astype(int)

    # Standardize continuous/numeric controls (z-score). Use population std (ddof=0) for interpretability.
    numeric_ctrls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_ctrls:
        # Coerce to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_ctrls)

    # z-score standardization
    for col in numeric_ctrls:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, create zero column
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = (df[col] - mean) / std

    # Interaction term between children and gender (moderation)
    df['Children_Gender'] = df['Children'] * df['GenderMale']

    # Final set of columns the model will use; keep original affairs as well
    keep_cols = [
        'affairs', 'AnyAffair', 'Children', 'GenderMale', 'Children_Gender',
        'age_z', 'yearsmarried_z', 'religiousness_z', 'education_z', 'occupation_z', 'rating_z'
    ]
    # Keep other columns if desired, but return at least keep_cols
    existing_keep = [c for c in keep_cols if c in df.columns]
    return df[existing_keep].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run two-part analysis to answer whether having children decreases engagement in extramarital affairs.

    1) Logistic regression for the probability of any affair (AnyAffair):
         AnyAffair ~ Children + GenderMale + Children_Gender + age_z + yearsmarried_z + religiousness_z + education_z + occupation_z + rating_z

    2) Count model for frequency among those with any affair (affairs > 0): negative binomial GLM with the same predictors.

    Returns a dictionary with fitted model result objects keyed by 'logit' and 'count_nb'.
    """
    results = {}
    df = df.copy()

    # Ensure required columns exist
    required = [
        'AnyAffair', 'affairs', 'Children', 'GenderMale', 'Children_Gender',
        'age_z', 'yearsmarried_z', 'religiousness_z', 'education_z', 'occupation_z', 'rating_z'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Prepare design matrix for logistic regression
    X_cols = ['Children', 'GenderMale', 'Children_Gender',
              'age_z', 'yearsmarried_z', 'religiousness_z', 'education_z', 'occupation_z', 'rating_z']
    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['AnyAffair']

    # Fit logistic regression (binary outcome)
    logit_model = sm.Logit(y, X)
    try:
        logit_res = logit_model.fit(disp=False)
    except Exception:
        # Fallback to robust method if convergence issues
        logit_res = logit_model.fit(disp=False, method='bfgs', maxiter=1000)

    results['logit'] = logit_res

    # Count model among respondents who report any affair (affairs > 0)
    df_count = df[df['affairs'] > 0].copy()
    if df_count.shape[0] < 10:
        # Too few positives to fit a reliable count model; return only logistic
        results['count_nb'] = None
        return results

    Xc = df_count[X_cols]
    Xc = sm.add_constant(Xc, has_constant='add')
    yc = df_count['affairs']

    # Fit Negative Binomial GLM (to handle overdispersion relative to Poisson)
    try:
        nb_model = sm.GLM(yc, Xc, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
    except Exception:
        # If GLM NegativeBinomial fails, fallback to Poisson with robust covariances
        pois_model = sm.GLM(yc, Xc, family=sm.families.Poisson())
        nb_res = pois_model.fit(cov_type='HC0')

    results['count_nb'] = nb_res

    # Return both fitted objects; caller can inspect summaries, params, conf_int, etc.
    return results


