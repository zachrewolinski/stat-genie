from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/shuffle_names_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe. The original provided schema has some column-description mismatches
    (e.g., 'culture' contains ages, 'age' contains a 0/1 order flag). This function standardizes names, creates derived variables,
    centers age, encodes binary indicators, and drops rows with missing values in key columns.

    Required output columns for the model: ['MajorityChosen', 'AgeYears', 'Age_c', 'Female', 'OrderMajorityFirst', 'CultureSite']
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Normalize column names that appear to be mis-labeled in the provided schema
    # Based on the schema values: 'culture' has values 4-14 -> this corresponds to age in years
    # 'age' has values 0/1 -> this corresponds to whether the majority was demonstrated first

    # Create AgeYears from 'culture'
    if 'culture' in df.columns:
        df['AgeYears'] = pd.to_numeric(df['culture'], errors='coerce')
    else:
        # if the expected column is missing, attempt to use 'age' (fallback)
        df['AgeYears'] = pd.to_numeric(df.get('age', pd.Series(dtype=float)), errors='coerce')

    # Create OrderMajorityFirst from 'age' (0/1 flag according to schema)
    if 'age' in df.columns:
        # Ensure the variable is binary 0/1
        df['OrderMajorityFirst'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)
    else:
        df['OrderMajorityFirst'] = 0

    # Map the choice outcome: majority_first (1=unchosen option, 2=majority option, 3=minority option)
    # Create a categorical label and a binary indicator for choosing majority
    if 'majority_first' in df.columns:
        df['ChoiceLabel'] = df['majority_first'].map({1: 'unchosen', 2: 'majority', 3: 'minority'})
        df['MajorityChosen'] = (df['majority_first'] == 2).astype(int)
    else:
        # If missing, create NA columns
        df['ChoiceLabel'] = pd.NA
        df['MajorityChosen'] = pd.NA

    # Recode gender: original encoding 1 = girl, 2 = boy. Create Female = 1 if girl else 0
    if 'gender' in df.columns:
        df['Female'] = df['gender'].apply(lambda x: 1 if pd.notna(x) and int(x) == 1 else 0).astype(int)
    else:
        df['Female'] = 0

    # Culture / site id: use 'y' if present (schema indicates 'y' is site ID 1..8)
    if 'y' in df.columns:
        # Make sure it's a string categorical variable for modeling
        df['CultureSite'] = 'site_' + df['y'].astype(int).astype(str)
    else:
        # If 'y' is missing, fallback to a generic single culture label
        df['CultureSite'] = 'site_unknown'

    # Drop rows with missing values in key columns for the planned analysis
    required = ['MajorityChosen', 'AgeYears', 'Female', 'OrderMajorityFirst', 'CultureSite']
    df = df.dropna(subset=required)

    # Center AgeYears for interpretability
    df['AgeYears'] = pd.to_numeric(df['AgeYears'], errors='coerce')
    mean_age = df['AgeYears'].mean()
    df['Age_c'] = df['AgeYears'] - mean_age

    # Ensure types
    df['MajorityChosen'] = df['MajorityChosen'].astype(int)
    df['OrderMajorityFirst'] = df['OrderMajorityFirst'].astype(int)
    df['Female'] = df['Female'].astype(int)
    df['CultureSite'] = df['CultureSite'].astype(str)

    # Keep only columns necessary for modeling (plus some originals for traceability)
    keep_cols = ['MajorityChosen', 'AgeYears', 'Age_c', 'Female', 'OrderMajorityFirst', 'CultureSite', 'ChoiceLabel', 'majority_first', 'gender', 'culture', 'age', 'y']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting the probability of choosing the majority option.
    The primary predictors are centered age (Age_c), CultureSite (categorical, included as a moderator via Age_c * C(CultureSite)),
    with controls for gender (Female) and demonstration order (OrderMajorityFirst).

    Returns the fitted GLMResults object.
    """
    import statsmodels.formula.api as smf
    # Ensure required columns are present
    required = ['MajorityChosen', 'Age_c', 'Female', 'OrderMajorityFirst', 'CultureSite']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula: allow Age effect to vary by culture via interaction Age_c * C(CultureSite)
    # This fits a main effect of Age_c and a separate age slope for each culture (via interactions).
    formula = 'MajorityChosen ~ Age_c * C(CultureSite) + Female + OrderMajorityFirst'

    # Fit GLM with binomial family (logistic regression)
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    results = model.fit()

    # It's often useful to inspect results.summary(), but the function returns the results object so the caller can use it.
    # print(results.summary())

    return results


