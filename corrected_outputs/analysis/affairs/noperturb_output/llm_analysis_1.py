from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/noperturb_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the Fair (Psychology Today) affairs dataset for modeling.
    - Drops rows with missing values in variables required for the analysis.
    - Creates a binary ChildrenBinary column (1 = 'yes', 0 = 'no').
    - Creates a gender_male column (1 = 'male', 0 = 'female').
    - Ensures numeric columns are numeric and affairs is integer.

    Returns the transformed dataframe containing at minimum the columns used in the model.
    """
    df = df.copy()

    # Columns needed for analysis
    required_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried',
                     'religiousness', 'education', 'occupation', 'rating']

    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Normalize string fields and map to binaries
    df['ChildrenBinary'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

    # Remove rows where mapping produced NA (unexpected values)
    df = df.dropna(subset=['ChildrenBinary', 'gender_male'])

    # Ensure numeric variables are numeric
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NA introduced by coercion
    df = df.dropna(subset=numeric_cols)

    # Ensure affairs is a non-negative integer count
    # (affairs in this dataset are coded as 0,1,2,3,7,12 etc.)
    df['affairs'] = df['affairs'].astype(int)
    df.loc[df['affairs'] < 0, 'affairs'] = 0

    # Keep only columns that will be used in modeling (but allow original columns to remain if desired)
    # Return full df (with added columns) so caller can inspect other fields if needed
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fits a count regression (Negative Binomial) predicting number of affairs.
    The model estimates the association of ChildrenBinary with affairs and tests
    whether this association differs by gender (ChildrenBinary * gender_male interaction).

    Formula:
      affairs ~ ChildrenBinary * gender_male + age + yearsmarried + religiousness + education + occupation + rating

    Returns the fitted model results object.
    """
    import statsmodels.formula.api as smf
    # Ensure the transformed columns exist
    required = ['affairs', 'ChildrenBinary', 'gender_male', 'age', 'yearsmarried',
                'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit Negative Binomial GLM to allow for overdispersion relative to Poisson
    formula = ('affairs ~ ChildrenBinary * gender_male + age + yearsmarried + '
               'religiousness + education + occupation + rating')

    nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()

    # Return the fitted results object (user can call .summary() or inspect params)
    return nb_model


