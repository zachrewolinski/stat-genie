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
    Transform the raw Fair dataset to prepare variables for modeling.

    Produces the following columns used by the model:
      - affairs: integer count outcome (keeps original values)
      - HasChildren: binary indicator 1 if 'children' == 'yes', 0 if 'no'
      - IsFemale: binary indicator 1 if gender == 'female', 0 if 'male'
      - age_c, yearsmarried_c, religiousness_c, education_c, occupation_c, rating_c: centered numeric controls

    Drops rows with missing values in any of the required columns.
    """
    df = df.copy()

    # Ensure expected columns are present
    required = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Keep a working subset to avoid surprises and drop rows with missing data in required columns
    df = df.loc[:, required].copy()
    df = df.dropna(subset=required)

    # Ensure affairs is integer (it may be numeric but with top-coding values like 7,12 already present)
    df['affairs'] = df['affairs'].astype(int)

    # Children -> HasChildren binary (1=yes, 0=no)
    # normalize text if necessary
    df['children'] = df['children'].astype(str).str.strip().str.lower()
    df['HasChildren'] = df['children'].map({'yes': 1, 'no': 0})
    # If any other values remain, treat as missing and drop
    df = df.dropna(subset=['HasChildren'])
    df['HasChildren'] = df['HasChildren'].astype(int)

    # Gender -> IsFemale binary (1=female, 0=male)
    df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    df['IsFemale'] = df['gender'].map({'female': 1, 'male': 0})
    df = df.dropna(subset=['IsFemale'])
    df['IsFemale'] = df['IsFemale'].astype(int)

    # Ensure numeric controls are numeric
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in numeric_cols:
        # coerce to numeric; drop if cannot convert
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    # Center numeric controls to aid interpretability
    df['age_c'] = df['age'] - df['age'].mean()
    df['yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()
    df['religiousness_c'] = df['religiousness'] - df['religiousness'].mean()
    df['education_c'] = df['education'] - df['education'].mean()
    df['occupation_c'] = df['occupation'] - df['occupation'].mean()
    df['rating_c'] = df['rating'] - df['rating'].mean()

    # Final column selection - keep the columns the model will use plus affairs
    final_cols = [
        'affairs',
        'HasChildren',
        'IsFemale',
        'age_c',
        'yearsmarried_c',
        'religiousness_c',
        'education_c',
        'occupation_c',
        'rating_c'
    ]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression model for affairs with main predictor HasChildren and controls.

    Primary specification: Negative Binomial (to account for overdispersion in the count outcome).
    Includes an interaction between HasChildren and IsFemale to test whether the effect of children
    differs by gender.

    Returns the fitted model results object.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Check required model columns exist
    required = ['affairs', 'HasChildren', 'IsFemale', 'age_c', 'yearsmarried_c', 'religiousness_c', 'education_c', 'occupation_c', 'rating_c']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for model: {missing}")

    # Build formula with interaction between HasChildren and IsFemale
    formula = (
        'affairs ~ HasChildren + IsFemale + HasChildren:IsFemale '
        '+ age_c + yearsmarried_c + religiousness_c + education_c + occupation_c + rating_c'
    )

    # Fit Negative Binomial GLM
    nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
    nb_results = nb_model.fit()

    # Provide basic diagnostics and return results object
    print(nb_results.summary())

    # As a robustness check, also fit a Poisson and compare AIC/BIC (optional)
    try:
        pois_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
        print('\nPoisson AIC:', pois_model.aic, 'NB AIC:', nb_results.aic)
    except Exception:
        pass

    return nb_results


