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
    Prepare the Fair (Psychology Today) dataset for modeling the effect of children on extramarital affairs.

    Transformations performed:
    - Copy input dataframe to avoid side effects.
    - Create numeric outcome column 'AffairCount' from 'affairs'.
    - Create binary independent variable 'HasChildren' from 'children' ('yes' -> 1, 'no' -> 0).
    - Create binary gender indicator 'Gender_Male' from 'gender' ('male' -> 1, 'female' -> 0).
    - Ensure numeric controls and create centered versions (suffix _c) for interpretability.
    - Drop rows with missing values in any columns required for the model.

    Final dataframe contains the columns used in the model: 
    ['AffairCount','HasChildren','Gender_Male','age_c','yearsmarried_c','religiousness_c','education_c','occupation_c','rating_c']
    """
    df = df.copy()

    # Outcome: make sure numeric
    df['AffairCount'] = pd.to_numeric(df.get('affairs'), errors='coerce')

    # Independent variable: children indicator
    # Accept variations in text; map yes/no to 1/0
    df['HasChildren'] = df.get('children').astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Gender dummy: male = 1, female = 0
    df['Gender_Male'] = df.get('gender').astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

    # Ensure numeric controls
    numeric_controls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_controls:
        # get and coerce to numeric
        df[col] = pd.to_numeric(df.get(col), errors='coerce')

    # Drop rows with missing values in any required modeling columns
    required = ['AffairCount', 'HasChildren', 'Gender_Male'] + numeric_controls
    df = df.dropna(subset=required)

    # Center continuous/numeric controls for interpretability (mean = 0)
    for col in numeric_controls:
        centered_col = f"{col}_c"
        df[centered_col] = df[col] - df[col].mean()

    # Keep only the columns needed for modeling plus original identifiers if present
    # (We keep all columns but ensure the required transformed columns exist.)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a regression model to estimate the association between having children and reported extramarital affairs,
    controlling for observed covariates. Because the outcome is a count-like variable with many zeros and
    potential over-dispersion, use a Negative Binomial generalized linear model (GLM).

    The model estimated:
      AffairCount ~ HasChildren + Gender_Male + HasChildren:Gender_Male + age_c + yearsmarried_c + 
                   religiousness_c + education_c + occupation_c + rating_c

    Returns the fitted model results object (statsmodels GLMResultsWrapper).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure the columns the transform step should have created are present
    required_cols = ['AffairCount', 'HasChildren', 'Gender_Male', 'age_c', 'yearsmarried_c',
                     'religiousness_c', 'education_c', 'occupation_c', 'rating_c']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula with interaction to allow effect of HasChildren to vary by gender
    formula = (
        'AffairCount ~ HasChildren + Gender_Male + HasChildren:Gender_Male '
        '+ age_c + yearsmarried_c + religiousness_c + education_c + occupation_c + rating_c'
    )

    # Fit Negative Binomial GLM
    # Note: this accounts for over-dispersion relative to Poisson; if NB fails consider Poisson or zero-inflated models.
    model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()

    # Return fitted model (caller can inspect .summary(), params, conf_int(), etc.)
    return model


