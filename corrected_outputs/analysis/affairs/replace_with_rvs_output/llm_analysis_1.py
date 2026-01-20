from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair affairs dataset into a modeling-ready dataframe.

    Steps performed:
    - Work on a copy of df.
    - Drop rows with missing values in the variables required for the analysis.
    - Encode 'children' into HasChildren (1=yes, 0=no).
    - Encode 'gender' into Gender_Male (1=male, 0=female).
    - Create standardized (z-scored) versions of continuous controls: age, yearsmarried, religiousness, education, occupation, rating.
    - Create an interaction term HasChildren_Gender = HasChildren * Gender_Male to test moderation by gender.
    - Return the cleaned/transformed dataframe containing all columns used in the model.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows with NA in core variables
    df = df.dropna(subset=required)

    # Encode HasChildren: map 'yes'->1, 'no'->0 (be robust to capitalization/whitespace)
    df['HasChildren'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # If mapping produced NaNs (unexpected category), drop those rows
    df = df[df['HasChildren'].notna()]
    df['HasChildren'] = df['HasChildren'].astype(int)

    # Encode Gender_Male: map 'male'->1, 'female'->0 (robust mapping)
    df['Gender_Male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    df = df[df['Gender_Male'].notna()]
    df['Gender_Male'] = df['Gender_Male'].astype(int)

    # Ensure affairs is numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Standardize continuous controls (z-scores). Use population std (ddof=0) to avoid small-sample issues.
    cont_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in cont_cols:
        s = pd.to_numeric(df[col], errors='coerce')
        mean = s.mean()
        std = s.std(ddof=0)
        # If std is zero (constant), set z to 0 to avoid division by zero
        if pd.isna(std) or std == 0:
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (s - mean) / std

    # Drop any rows that still have missing values in the modeling columns
    model_cols = ['affairs', 'HasChildren', 'Gender_Male'] + [c + '_z' for c in cont_cols]
    df = df.dropna(subset=model_cols)

    # Interaction term for moderation test
    df['HasChildren_Gender'] = df['HasChildren'].astype(int) * df['Gender_Male'].astype(int)

    # Keep only columns required for modeling plus originals for traceability
    keep_cols = list(set(model_cols + ['HasChildren_Gender']))
    # Keep also the original children and gender columns for reference
    keep_cols += ['children', 'gender']
    # Preserve column order for readability
    keep_cols = [c for c in ['affairs', 'HasChildren', 'Gender_Male', 'HasChildren_Gender'] + [c + '_z' for c in cont_cols] + ['children', 'gender'] if c in keep_cols]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression to answer whether having children decreases engagement in extramarital affairs.

    Model specification:
    - Dependent variable: affairs (count)
    - Main independent variable: HasChildren
    - Moderator: Gender_Male (interaction HasChildren_Gender included)
    - Controls: standardized age, years married, religiousness, education, occupation, rating
    - Estimator: Negative Binomial (GLM) to account for over-dispersion in count data

    Returns the fitted statsmodels object (results) so the caller can inspect summary, params, conf_int, etc.
    """
    import statsmodels.api as sm

    # Required columns for the model
    exog_cols = [
        'HasChildren',
        'Gender_Male',
        'HasChildren_Gender',
        'age_z',
        'yearsmarried_z',
        'religiousness_z',
        'education_z',
        'occupation_z',
        'rating_z'
    ]

    missing = [c for c in exog_cols + ['affairs'] if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    X = df[exog_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['affairs'].astype(float)

    # Fit Negative Binomial GLM
    # Use statsmodels' GLM with NegativeBinomial family to allow for overdispersion relative to Poisson
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = model.fit()

    # Return the fitted results object so the caller can inspect .summary(), .params, etc.
    return results


