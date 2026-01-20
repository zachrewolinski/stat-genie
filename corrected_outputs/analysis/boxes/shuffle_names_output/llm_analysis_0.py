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
    Transform the raw dataset into a dataframe suitable for modeling.

    Expected original columns (per provided schema):
      - 'majority_first': numeric code for choice (1 = unchosen option, 2 = majority option, 3 = minority option)
      - 'gender': 1 = girl, 2 = boy
      - 'culture': (mislabelled in schema) contains child's age in years (e.g., 4-14)
      - 'age': (mislabelled in schema) contains indicator whether majority was demonstrated first (0/1)
      - 'y': site ID (1..8) — treated as culture / site identifier

    The function will:
      - drop rows with missing values in required columns
      - create Age (continuous), Culture (categorical), MajorityChoice (binary), IsMale (binary), MajorityDemoFirst (binary)
      - return a dataframe that contains at least the columns used in the statistical model
    """
    # Copy to avoid modifying original
    df = df.copy()

    # Ensure required columns exist
    required = ['majority_first', 'gender', 'culture', 'age', 'y']
    missing_cols = [c for c in required if c not in df.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing required columns for transform: {missing_cols}")

    # Drop rows with missing values in the columns we need for modeling
    df = df.dropna(subset=['majority_first', 'gender', 'culture', 'age', 'y'])

    # Build Age from the 'culture' column (schema indicates this column actually contains age in years)
    # Convert to numeric (float) in case of string input
    df['Age'] = pd.to_numeric(df['culture'], errors='coerce')

    # Build binary outcome: MajorityChoice = 1 if child picked the majority option (majority_first == 2), else 0
    # If codes are strings, coerce to numeric first
    df['majority_first_num'] = pd.to_numeric(df['majority_first'], errors='coerce')
    df['MajorityChoice'] = (df['majority_first_num'] == 2).astype(int)
    df = df.drop(columns=['majority_first_num'])

    # Gender -> IsMale (1 = boy, 0 = girl). Coerce to numeric then map
    df['gender_num'] = pd.to_numeric(df['gender'], errors='coerce')
    # Map: 2 => boy, 1 => girl (per schema). Anything else becomes NaN and will be dropped below
    df['IsMale'] = df['gender_num'].map({2: 1, 1: 0})
    df = df.drop(columns=['gender_num'])

    # MajorityDemoFirst: the 'age' column in the provided schema actually indicates whether the majority
    # option was demonstrated first (0/1). Coerce to numeric and binarize.
    df['MajorityDemoFirst'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')

    # Culture: create a categorical site identifier from 'y' (site id). Keep as category for modeling.
    df['Culture'] = pd.Categorical(df['y'])

    # Drop any rows where conversions created missing values in the modeling columns
    model_cols = ['MajorityChoice', 'Age', 'Culture', 'IsMale', 'MajorityDemoFirst']
    df = df.dropna(subset=model_cols)

    # Ensure types: Age float, MajorityChoice/IsMale/MajorityDemoFirst integer, Culture category
    df['Age'] = df['Age'].astype(float)
    df['MajorityChoice'] = df['MajorityChoice'].astype(int)
    df['IsMale'] = df['IsMale'].astype(int)
    # MajorityDemoFirst may be Int64 NA-capable; cast to int after NA removal
    df['MajorityDemoFirst'] = df['MajorityDemoFirst'].astype(int)

    # Keep only the columns necessary for modeling (but preserve original data if desired by user)
    # Here we return the full df but ensure the modeling columns exist.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a logistic regression (GLM with binomial family) predicting the probability a child chooses the majority option.

    Model specification:
      MajorityChoice ~ Age * C(Culture) + IsMale + MajorityDemoFirst

    This specification tests:
      - main effect of Age (developmental change in majority reliance)
      - main effect of Culture (differences in baseline majority reliance across sites)
      - Age x Culture interaction (whether developmental slopes differ across cultures)
      - controls for child's gender and whether the majority was demonstrated first

    Returns the fitted statsmodels GLMResults object.
    """
    import statsmodels.formula.api as smf

    # Verify required columns present
    req = ['MajorityChoice', 'Age', 'Culture', 'IsMale', 'MajorityDemoFirst']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for modeling: {missing}")

    # Fit the binomial GLM with categorical Culture and an Age x Culture interaction
    formula = 'MajorityChoice ~ Age * C(Culture) + IsMale + MajorityDemoFirst'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model object (user can call .summary(), .params, .predict(), etc.)
    return model


