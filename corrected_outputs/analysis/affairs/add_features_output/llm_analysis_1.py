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
    Transform the raw Fair affairs dataframe into a dataframe ready for modeling.

    Produces the following columns required by the model:
      - affairs_count: numeric count of extramarital affairs (from 'affairs')
      - HasChildren: binary indicator (1 if 'children' == 'yes', 0 if 'no')
      - Female: binary indicator for female gender (1 if gender indicates female, else 0)
      - HasChildren_Female: interaction term HasChildren * Female
      - age, yearsmarried, religiousness, education, occupation, rating: numeric controls

    Drops rows with missing values in any of the model variables.
    """
    # Work on a copy
    df = df.copy()

    # Standardize/clean affairs -> integer count
    df['affairs_count'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Children: map common textual values to binary 1=yes, 0=no
    # If values already 0/1 or booleans, handle gracefully
    def _map_children(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            # if numeric but encoded as 0/1 or 0/1 strings cast
            return int(x) if x in (0, 1) else (1 if x else 0)
        s = str(x).strip().lower()
        if s in ('yes', 'y', '1', 'true', 't'):
            return 1
        if s in ('no', 'n', '0', 'false', 'f'):
            return 0
        return np.nan

    df['HasChildren'] = df['children'].apply(_map_children)

    # Gender -> Female binary indicator
    def _map_female(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        # common encodings 'female'/'male', 'f'/'m'
        if s.startswith('f'):
            return 1
        if s.startswith('m'):
            return 0
        # fallback: try to interpret numeric 0/1
        try:
            xv = float(x)
            return 1 if xv == 1 else 0
        except Exception:
            return np.nan

    df['Female'] = df['gender'].apply(_map_female)

    # Ensure numeric controls are numeric, coerce errors to NaN
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # if a control is missing from the dataframe, create as NaN so later drop will remove rows
            df[col] = np.nan

    # Interaction term for moderation (HasChildren * Female)
    df['HasChildren_Female'] = df['HasChildren'] * df['Female']

    # Drop any rows with missing values in the variables we will use in the model
    required_cols = ['affairs_count', 'HasChildren', 'Female', 'HasChildren_Female'] + numeric_cols
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Ensure affairs_count is integer (counts)
    df['affairs_count'] = df['affairs_count'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model appropriate for a non-negative, zero-inflated dependent variable.

    We fit a Zero-Inflated Negative Binomial (ZINB) model using statsmodels.
    The count equation includes HasChildren, Female, HasChildren_Female and other numeric controls.
    The inflation (zero) equation uses a subset (HasChildren and Female) so the model can account
    for excess zeros possibly related to children and gender.

    Returns the fitted model results object.
    """
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Select exogenous columns for the count model
    exog_cols = [
        'HasChildren',
        'Female',
        'HasChildren_Female',
        'age',
        'yearsmarried',
        'religiousness',
        'education',
        'occupation',
        'rating'
    ]

    # Ensure required columns exist
    missing = [c for c in exog_cols + ['affairs_count'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Add constant to exog (count) and to inflation exog
    exog = sm.add_constant(df[exog_cols], has_constant='add')
    # Use simpler inflation model with HasChildren and Female (plus constant)
    exog_infl = sm.add_constant(df[['HasChildren', 'Female']], has_constant='add')

    endog = df['affairs_count'].astype(int)

    # Fit ZINB model. p=1 uses the standard NB parameterization. We suppress convergence output.
    model = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, p=1)

    # Fit with robust starting params if default struggles. Use method='newton' for reliability.
    try:
        results = model.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        # fallback to default fit settings
        results = model.fit(disp=False)

    # Return the fitted results object. The caller can inspect summary(), params, conf_int(), etc.
    return results


