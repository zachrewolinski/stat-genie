from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following final columns used in the model:
      - MajorityChoice: binary DV (1 if y==2 [majority], else 0)
      - Age_c: centered age (age - mean(age))
      - Culture: categorical site identifier (from original 'culture')
      - Gender_Boy: binary (1 if gender==2 (boy), 0 if gender==1 (girl))
      - MajorityFirst: binary copy of the 'majority_first' column

    The function drops rows with missing values in the variables required for modeling.
    """
    df = df.copy()

    # Ensure necessary columns exist
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing outcome or key predictors/controls
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Create binary outcome: 1 if majority option (y == 2), else 0
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Center age for interpretability in interaction terms
    df['Age_c'] = df['age'] - df['age'].mean()

    # Convert culture to a categorical variable for modeling (keeps original codes but marks as category)
    df['Culture'] = df['culture'].astype('category')

    # Create gender binary: original coding 1=girl, 2=boy. Create Boy=1 indicator
    df['Gender_Boy'] = df['gender'].map({1: 0, 2: 1})
    # If any unexpected gender codes appear, coerce to 0 and warn
    if df['Gender_Boy'].isna().any():
        df['Gender_Boy'] = df['Gender_Boy'].fillna(0).astype(int)
    else:
        df['Gender_Boy'] = df['Gender_Boy'].astype(int)

    # Ensure majority_first is binary integer (0/1)
    df['MajorityFirst'] = df['majority_first'].astype(int)

    # Keep only columns needed for modeling (plus optionally original columns if desired)
    cols_to_keep = ['MajorityChoice', 'Age_c', 'Culture', 'Gender_Boy', 'MajorityFirst', 'y', 'age', 'culture', 'gender', 'majority_first']
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[existing_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a logistic regression predicting the probability of choosing the majority option.

    Model specification:
      MajorityChoice ~ Age_c * C(Culture) + Gender_Boy + MajorityFirst

    - Age_c * C(Culture) tests whether the age slope differs across cultural sites (i.e., whether developmental trajectories of majority reliance vary by culture).
    - Gender_Boy and MajorityFirst are included as controls.

    Returns the fitted statsmodels logit result object. Also computes average marginal effects of Age by Culture for interpretability.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Make a copy to avoid modifying the input
    df = df.copy()

    # Basic sanity checks
    required = ['MajorityChoice', 'Age_c', 'Culture', 'Gender_Boy', 'MajorityFirst']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit logistic regression with culture fixed effects and Age x Culture interactions
    formula = 'MajorityChoice ~ Age_c * C(Culture) + Gender_Boy + MajorityFirst'
    logit_model = smf.logit(formula=formula, data=df)
    results = logit_model.fit(disp=False)

    # Compute average marginal effect of Age overall and (optionally) by culture.
    # Overall marginal effect for Age_c
    try:
        margeff_overall = results.get_margeff(at='overall', method='dydx', atexog=None).summary_frame()
    except Exception:
        margeff_overall = None

    # Package results into a dictionary for convenient downstream use
    out = {
        'model_result': results,
        'marginal_effects_overall': margeff_overall
    }

    return out


