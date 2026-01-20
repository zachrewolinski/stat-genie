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
    Transform the raw Fair (1978) affairs dataset into a dataframe ready for modeling.

    Outputs (added columns used in the model):
      - AffairCount: integer count of extramarital acts (from 'affairs')
      - HasChildren: binary indicator 1=yes 0=no (from 'children')
      - Female: binary indicator 1=female 0=male (from 'gender')
      - HasChildren_Female: interaction term HasChildren * Female
      - *_c: standardized (z-scored) versions of continuous controls: age_c, yearsmarried_c,
             religiousness_c, education_c, occupation_c, rating_c

    The function also drops rows with missing values in any of the variables required for the model.
    """
    # Make a copy to avoid modifying caller's frame
    df = df.copy()

    # Ensure 'affairs' is numeric and create AffairCount as integer counts
    # Some datasets encode ranges with numeric codes already; we preserve those numeric codes as counts
    df['AffairCount'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Map children to binary HasChildren (yes -> 1, no -> 0)
    df['HasChildren'] = df['children'].map({'yes': 1, 'no': 0})

    # Map gender to binary Female (female -> 1, male -> 0)
    # Accept common variants by lower-casing
    df['gender_clean'] = df['gender'].astype(str).str.lower()
    df['Female'] = df['gender_clean'].map({'female': 1, 'male': 0})
    df.drop(columns=['gender_clean'], inplace=True)

    # Interaction term (to allow different HasChildren effect by gender)
    df['HasChildren_Female'] = df['HasChildren'] * df['Female']

    # Continuous control variables to standardize (z-score). If a column isn't numeric, coerce and create NA.
    cont_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in cont_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute standardized versions (mean 0, sd 1). If sd is zero or NA, leave as NA.
    for col in cont_cols:
        mean = df[col].mean()
        std = df[col].std()
        col_z = f"{col}_c"
        if pd.isna(std) or std == 0:
            # fallback: subtract mean only
            df[col_z] = df[col] - mean
        else:
            df[col_z] = (df[col] - mean) / std

    # Select the columns that the model will require and drop rows with missing data in any of them
    required_cols = [
        'AffairCount', 'HasChildren', 'Female', 'HasChildren_Female',
        'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c'
    ]

    # Drop rows where any required column is NA
    df = df.dropna(subset=required_cols)

    # Ensure AffairCount is integer-valued (counts). Keep as numeric dtype for modeling.
    df['AffairCount'] = df['AffairCount'].astype(float)

    # Return the transformed dataframe (contains all original columns plus the new ones)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a Negative Binomial regression of AffairCount on HasChildren (primary predictor),
    controlling for demographic and marriage-related covariates, and including an interaction
    with gender to test moderation.

    Model specification:
      AffairCount ~ HasChildren + Female + HasChildren_Female + age_c + yearsmarried_c
                   + religiousness_c + education_c + occupation_c + rating_c

    Returns the fitted results object (statsmodels GLMResults).
    """
    # Columns used as predictors
    exog_cols = [
        'HasChildren', 'Female', 'HasChildren_Female',
        'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c'
    ]

    # Prepare design matrices
    X = df[exog_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['AffairCount']

    # Fit a Negative Binomial GLM to allow for overdispersion in count data.
    # If NegativeBinomial() isn't available in the user's environment, an alternative is
    # to use sm.GLM with family=sm.families.Poisson() and check dispersion or use discrete NegativeBinomial.
    model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = model_nb.fit()

    # Return the fitted results object so the caller can inspect params, summary, etc.
    return results


