from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/shuffle_names_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe. Assumptions based on schema:
      - 'majority_first': outcome (1=unchosen option, 2=majority option, 3=minority option)
      - 'culture': actually contains child's age in years (4-14)
      - 'age': binary indicator for whether majority was demonstrated first (0/1)
      - 'y': site ID (1..8)
      - 'gender': 1=girl, 2=boy

    Produces columns used in modeling:
      - is_majority_choice: binary DV (1 if majority chosen, 0 otherwise)
      - age_years: raw age in years (from 'culture')
      - age_c: mean-centered age used in models
      - site_id: categorical site identifier (from 'y')
      - is_boy: binary control for gender (1=boy, 0=girl)
      - demo_order_majority_first: binary control (from 'age' column in input)
    """
    # Work on a copy
    df = df.copy()

    # Drop rows missing any variables needed for analysis
    required_cols = ['majority_first', 'culture', 'age', 'y', 'gender']
    df = df.dropna(subset=required_cols)

    # DV: whether child chose the majority option (majority_first == 2)
    df['is_majority_choice'] = (df['majority_first'] == 2).astype(int)

    # Age: the field named 'culture' in the provided schema contains 4-14 -> treat as age
    df['age_years'] = pd.to_numeric(df['culture'], errors='coerce')

    # Mean-center age for interpretability and to stabilize interactions
    df['age_c'] = df['age_years'] - df['age_years'].mean()

    # Site / cultural context: use 'y' as site id
    df['site_id'] = df['y'].astype('category')

    # Gender: convert to binary indicator for boy (1) vs girl (0). Input: 1=girl, 2=boy
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Demonstration order: input column 'age' encodes whether majority was demonstrated first (0/1)
    # Ensure it's binary integer
    df['demo_order_majority_first'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)

    # Keep only the columns required for analysis (and any originals for traceability)
    keep_cols = ['is_majority_choice', 'age_years', 'age_c', 'site_id', 'is_boy', 'demo_order_majority_first',
                 'majority_first', 'culture', 'age', 'y', 'gender']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic (binomial) regression testing how probability of choosing the majority option
    changes with age, and whether that age-effect differs across cultural contexts (site_id).

    Model (GLM, binomial link):
      is_majority_choice ~ age_c * C(site_id) + is_boy + demo_order_majority_first

    - age_c * C(site_id) lets each site have its own age slope (tests the research question: how
      reliance on majority preference develops with age across cultural contexts).
    - We include is_boy and demo_order_majority_first as covariates.

    Returns the fitted model object (statsmodels GLMResults) so the caller can inspect coefficients,
    confidence intervals, predictions, etc.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure necessary columns exist
    required = ['is_majority_choice', 'age_c', 'site_id', 'is_boy', 'demo_order_majority_first']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit binomial GLM with logit link
    # Note: C(site_id) treats site_id as categorical fixed effect; interaction with age_c allows site-specific age slopes
    formula = 'is_majority_choice ~ age_c * C(site_id) + is_boy + demo_order_majority_first'
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted model results object for downstream inspection (summary, params, conf_int, predict, etc.)
    return model


