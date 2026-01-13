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
    Transform the raw Fair (affairs) dataset into a modeling dataframe.

    Outputs (added columns):
      - AffairCount: numeric copy of 'affairs' (count outcome)
      - AnyAffair: binary indicator (1 if AffairCount > 0, else 0)
      - HasChildren: binary indicator derived from 'children' (1 = yes, 0 = no)
      - IsFemale: binary indicator (1 = female, 0 = male) derived from 'gender'
      - Age_z, YearsMarried_z, Religiousness_z, Education_z, Occupation_z, Rating_z: z-scored continuous controls

    Rows with missing data in any variable required for modeling are dropped.
    """
    df = df.copy()

    # Ensure the columns we will use exist; if they do not, this will raise a KeyError so the caller knows.
    required_original_cols = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]

    # Drop rows with missing values in any of these required columns
    df = df.dropna(subset=required_original_cols)

    # Dependent variable: Affair count
    # Keep original numeric coding; convert to numeric to be safe
    df['AffairCount'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Binary indicator for any affair
    df['AnyAffair'] = (df['AffairCount'] > 0).astype(int)

    # Independent variable: HasChildren (1 if 'yes', 0 if 'no')
    # Normalize strings and map
    df['HasChildren'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Some datasets might encode children as booleans or 0/1 already; handle numeric
    if df['HasChildren'].isnull().any():
        # Try numeric conversion for remaining values
        try:
            num_children_map = pd.to_numeric(df.loc[df['HasChildren'].isnull(), 'children'], errors='coerce')
            # If numeric and >0 treat as having children
            df.loc[df['HasChildren'].isnull(), 'HasChildren'] = (num_children_map > 0).astype(float)
        except Exception:
            pass

    # Gender -> IsFemale (1 female, 0 male)
    df['IsFemale'] = df['gender'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})

    # Standardize (z-score) continuous controls. Use population std (ddof=0).
    def zscore(s: pd.Series, name: str) -> pd.Series:
        s_num = pd.to_numeric(s, errors='coerce')
        mean = s_num.mean()
        std = s_num.std(ddof=0)
        if pd.isna(std) or std == 0:
            return (s_num - mean).fillna(0)
        return ((s_num - mean) / std).fillna(0)

    df['Age_z'] = zscore(df['age'], 'age')
    df['YearsMarried_z'] = zscore(df['yearsmarried'], 'yearsmarried')
    df['Religiousness_z'] = zscore(df['religiousness'], 'religiousness')
    df['Education_z'] = zscore(df['education'], 'education')
    df['Occupation_z'] = zscore(df['occupation'], 'occupation')
    df['Rating_z'] = zscore(df['rating'], 'rating')

    # After deriving, drop any rows that still have NA in key derived columns
    model_cols = [
        'AffairCount', 'AnyAffair', 'HasChildren', 'IsFemale',
        'Age_z', 'YearsMarried_z', 'Religiousness_z', 'Education_z',
        'Occupation_z', 'Rating_z'
    ]
    df = df.dropna(subset=model_cols)

    # Ensure correct dtypes
    integer_cols = ['AnyAffair', 'HasChildren', 'IsFemale']
    for c in integer_cols:
        df[c] = df[c].astype(int)

    df['AffairCount'] = df['AffairCount'].astype(float)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to assess whether having children is associated with fewer extramarital affairs:
      1) Logistic regression for the probability of any affair (AnyAffair)
      2) Negative binomial regression for the count of affairs (AffairCount)

    The function returns a dict with fitted model results objects.
    """
    # Copy to avoid modifying original
    d = df.copy()

    # Define predictors
    predictors = [
        'HasChildren', 'IsFemale', 'Age_z', 'YearsMarried_z',
        'Religiousness_z', 'Education_z', 'Occupation_z', 'Rating_z'
    ]

    # Prepare X and add constant
    X = sm.add_constant(d[predictors], has_constant='add')

    results = {}

    # 1) Logistic regression: probability of any affair (binary outcome)
    try:
        logit_model = sm.Logit(d['AnyAffair'], X)
        logit_res = logit_model.fit(disp=False)
        results['logit'] = logit_res
    except Exception as e:
        # If Logit fails (e.g., perfect separation), capture the exception
        results['logit_error'] = str(e)

    # 2) Negative binomial regression for the count outcome
    # Use GLM with NegativeBinomial family; this handles over-dispersion vs Poisson.
    try:
        nb_model = sm.GLM(d['AffairCount'], X, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
        results['neg_bin'] = nb_res
    except Exception as e:
        results['neg_bin_error'] = str(e)

    # Return fitted result objects (or error messages) so the caller can inspect summary, params, etc.
    return results


