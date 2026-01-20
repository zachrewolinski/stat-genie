from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair affairs dataset into the analysis dataframe.

    Produces columns used by the model:
      - affairs: numeric outcome, cleaned
      - Children: binary indicator 1=yes, 0=no
      - gender_male: binary indicator 1=male, 0=female
      - age, yearsmarried, religiousness, education, occupation, rating: numeric controls

    Rows with missing values in the outcome or main predictors/controls are dropped.
    """
    df = df.copy()

    # Ensure affairs numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Normalize children to binary indicator
    # Accepts 'yes'/'no' (case insensitive) or already-coded forms
    df['Children'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    # If mapping produced NaN but there are 0/1 values already, try to coerce
    if df['Children'].isna().any():
        # Attempt numeric coercion for those that failed
        df.loc[df['Children'].isna(), 'Children'] = pd.to_numeric(df.loc[df['Children'].isna(), 'children'], errors='coerce')

    # Create gender_male indicator: 1 if male, 0 if female
    df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map(lambda x: 1 if x == 'male' else (0 if x == 'female' else np.nan))

    # Ensure control variables are numeric
    numeric_controls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_controls:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # if a control is missing from the dataset, create as NaN to make missingness explicit
            df[col] = np.nan

    # Keep only rows with non-missing outcome and main independent variable and core controls
    required = ['affairs', 'Children', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required)

    # Ensure affairs is non-negative integer-like; round if necessary
    df['affairs'] = df['affairs'].clip(lower=0)
    # Some values in the original dataset are top-coded indicators (e.g., 7, 12). Keep as reported.
    # Cast to integer if they are effectively integer values
    if (df['affairs'] % 1 == 0).all():
        df['affairs'] = df['affairs'].astype(int)

    # Final dataframe contains the columns needed for modeling
    keep_cols = ['affairs', 'Children', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df[keep_cols].reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression model to estimate the effect of having children on extramarital affairs.

    Primary specification: Negative Binomial regression controlling for gender, age, years married,
    religiousness, education, occupation, and marital rating. Robust (HC0) standard errors are returned.

    Returns the fitted model results object.
    """
    import statsmodels.formula.api as smf
    # Formula: affairs as a function of Children and controls
    formula = 'affairs ~ Children + gender_male + age + yearsmarried + religiousness + education + occupation + rating'

    # Fit a GLM with Negative Binomial family (handles overdispersion relative to Poisson)
    # Use robust (HC0) covariance for standard errors
    model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC0')

    # Print summary for convenience; return results for programmatic use
    print(model.summary())
    return model


