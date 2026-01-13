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
    Prepare the Fair (affairs) dataset for analysis.

    Creates the following columns used in modeling (exact names):
      - AffairsCount: numeric copy of original 'affairs' column
      - AnyAffair: binary indicator (1 if AffairsCount > 0, else 0)
      - HasChildren: binary indicator (1 if 'children' == 'yes', 0 if 'no')
      - IsMale: binary indicator (1 if gender == 'male', 0 otherwise)
      - Age, YearsMarried, Religiousness, Education, Occupation, Rating: numeric controls

    Drops rows with missing values in any variable needed for the primary analysis.
    """
    df = df.copy()

    # Keep relevant columns and ensure they exist
    needed = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")

    # Convert/clean columns
    # AffairsCount: preserve numeric coding as provided (0,1,2,3,7,12 etc.)
    df['AffairsCount'] = pd.to_numeric(df['affairs'], errors='coerce')

    # HasChildren: map 'yes'/'no' to 1/0; support mixed capitalization
    df['HasChildren'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Binary indicator for any affair
    df['AnyAffair'] = (df['AffairsCount'] > 0).astype(int)

    # Gender indicator: IsMale = 1 if 'male', 0 otherwise
    df['IsMale'] = df['gender'].astype(str).str.strip().str.lower().map(lambda x: 1 if x == 'male' else 0)

    # Numeric controls: coerce to numeric, keep original variable meaning
    df['Age'] = pd.to_numeric(df['age'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['Religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['Education'] = pd.to_numeric(df['education'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows with missing values in any of the model columns
    model_cols = ['AffairsCount', 'AnyAffair', 'HasChildren', 'IsMale', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'Rating']
    df = df.dropna(subset=model_cols)

    # Ensure integer types where appropriate
    df['HasChildren'] = df['HasChildren'].astype(int)
    df['IsMale'] = df['IsMale'].astype(int)
    df['AnyAffair'] = df['AnyAffair'].astype(int)

    # Return transformed dataframe containing at least the columns listed in cvars
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs two complementary models to assess whether having children decreases engagement in extramarital affairs.

    1) Primary analysis: logistic regression (GLM Binomial) on AnyAffair (binary outcome).
       - Includes HasChildren as primary predictor and controls (IsMale, Age, YearsMarried,
         Religiousness, Education, Occupation, Rating).
       - Also fits a second logistic model that includes the interaction HasChildren * IsMale to test
         whether the effect of children differs by gender.

    2) Robustness / secondary analysis: Negative binomial regression (GLM NegativeBinomial)
       on AffairsCount (count outcome with top-coding) using the same covariates and interaction.

    Returns a dictionary with fitted results objects from statsmodels for each model.
    """
    # copy to avoid side effects
    df = df.copy()

    # Define covariates used in all models
    covariates = ['HasChildren', 'IsMale', 'Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'Rating']

    # Prepare design matrix (no categorical expansion beyond IsMale which we've coded)
    X = df[covariates]
    X = sm.add_constant(X)

    # Outcome: binary AnyAffair
    y_bin = df['AnyAffair']

    # Fit logistic (GLM with Binomial family)
    logit_model = sm.GLM(y_bin, X, family=sm.families.Binomial()).fit()

    # Fit logistic with interaction HasChildren * IsMale
    X_int = X.copy()
    X_int['HasChildren_x_IsMale'] = X_int['HasChildren'] * X_int['IsMale']
    logit_model_inter = sm.GLM(y_bin, X_int, family=sm.families.Binomial()).fit()

    # Secondary: Negative binomial on count outcome (AffairsCount) as robustness
    y_count = df['AffairsCount']
    # Use same X_int (with interaction) for count model
    nb_model = sm.GLM(y_count, X_int, family=sm.families.NegativeBinomial()).fit()

    # Package results
    results = {
        'logit': logit_model,
        'logit_interaction': logit_model_inter,
        'negbin': nb_model
    }

    return results


