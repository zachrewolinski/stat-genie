from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/boxes/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe with the exact columns used in the model.

    Expected original columns (per dataset schema):
      - 'majority_first' : outcome (1=unchosen option, 2=majority option, 3=minority option)
      - 'gender'        : 1 = girl, 2 = boy
      - 'culture'       : (misnamed) contains child's age in years (4-14)
      - 'age'           : (misnamed) contains a demonstration-order flag (0/1) indicating whether majority was shown first
      - 'y'             : site id (1-8) indicating cultural context

    The function will:
      - Drop rows with missing key fields
      - Create MajorityChoice (binary), Age, Age_centered, Site, Gender, MajorityOrder
      - Ensure column dtypes are appropriate
    """
    df = df.copy()

    # Keep only relevant columns if present
    required_cols = ['majority_first', 'gender', 'culture', 'age', 'y']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Convert to numeric where appropriate (coerce errors to NaN)
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').astype('Int64')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce').astype('Int64')
    df['culture'] = pd.to_numeric(df['culture'], errors='coerce')  # this column encodes age in years
    df['age'] = pd.to_numeric(df['age'], errors='coerce')         # this column encodes demonstration-order flag
    df['y'] = pd.to_numeric(df['y'], errors='coerce').astype('Int64')

    # Drop rows with missing core variables
    df = df.dropna(subset=['majority_first', 'gender', 'culture', 'age', 'y'])

    # Create binary dependent variable: MajorityChoice (1 if chose majority option, else 0)
    df['MajorityChoice'] = (df['majority_first'] == 2).astype(int)

    # Create Age (in years) and a centered age variable for modeling
    # According to the dataset schema, 'culture' actually contains age in years
    df['Age'] = df['culture'].astype(float)
    # Optionally filter implausible ages (keep typical study range 3-16 to be permissive)
    df = df[(df['Age'] >= 3) & (df['Age'] <= 16)]
    df['Age_centered'] = df['Age'] - df['Age'].mean()

    # Create Site (categorical culture / site identifier) from 'y'
    # Represent as simple integer category; model will treat it as categorical via C(Site)
    df['Site'] = df['y'].astype(int)

    # Map gender to a simple coded column; leave numeric (1/2) and model as categorical
    df['Gender'] = df['gender'].astype(int)

    # MajorityOrder: whether majority was demonstrated first
    # Per schema 'age' is the demonstration-order flag (0/1). We'll coerce to int 0/1.
    df['MajorityOrder'] = df['age'].astype(int)

    # Keep only the columns needed for modeling plus helpful originals
    final_cols = ['MajorityChoice', 'Age', 'Age_centered', 'Site', 'Gender', 'MajorityOrder',
                  'majority_first', 'culture', 'age', 'y']
    # Some of these may duplicate; ensure they exist then subset
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) regression to test how reliance on the majority develops with age across sites.

    Model specification (fixed-effects GLM with binomial family):
      MajorityChoice ~ Age_centered * C(Site) + C(Gender) + MajorityOrder

    - The Age_centered * C(Site) interaction allows the slope of age to vary across cultural sites (different developmental trajectories).
    - Gender and demonstration-order (MajorityOrder) are included as covariates.

    Returns the fitted GLMResults object.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Verify required columns exist
    required = ['MajorityChoice', 'Age_centered', 'Site', 'Gender', 'MajorityOrder']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Fit binomial GLM (logistic regression) with site as categorical and interaction with age
    formula = 'MajorityChoice ~ Age_centered * C(Site) + C(Gender) + MajorityOrder'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model object (caller can inspect model.summary(), params, conf_int, etc.)
    return model


