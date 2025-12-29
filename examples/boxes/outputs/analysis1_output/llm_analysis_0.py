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
    Transform the original dataframe to the analysis dataframe.

    Steps:
    - Drop rows with missing values in variables required for the model.
    - Create binary outcome MajorityChoice: 1 if y==2 (majority), 0 otherwise.
    - Create binary gender indicator IsGirl (1 if gender==1, per data dictionary: 1=girl, 2=boy).
    - Center age (Age_c) and create quadratic term Age_c_sq for non-linear effects.
    - Ensure culture is categorical.

    Returns the dataframe with new columns used in modeling.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required columns for the planned model
    required_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    # Drop rows with NA in any required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')
    df['culture'] = df['culture'].astype('category')

    # Re-drop if conversion produced NA
    df = df.dropna(subset=required_cols)

    # Dependent variable: majority choice (1 if y==2, else 0)
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Control: IsGirl (1 if girl, 0 if boy). According to data: 1=girl, 2=boy
    df['IsGirl'] = (df['gender'] == 1).astype(int)

    # Center age and add quadratic term
    df['Age_c'] = df['age'] - df['age'].mean()
    df['Age_c_sq'] = df['Age_c'] ** 2

    # Ensure majority_first is 0/1 integer
    df['majority_first'] = df['majority_first'].astype(int)

    # Final sanity check: keep rows with finite Age_c and MajorityChoice
    df = df[df['MajorityChoice'].notnull()]
    df = df[np.isfinite(df['Age_c'])]

    # Reset index for downstream modeling convenience
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a logistic regression to model the probability of choosing the majority option.

    Model specification (fixed-effects):
      MajorityChoice ~ Age_c + Age_c_sq + IsGirl + majority_first + C(culture) + Age_c:C(culture)

    - C(culture) adds a set of culture-specific intercepts (baseline differences across sites).
    - Age_c:C(culture) fits culture-specific linear age slopes (allows developmental trajectories to differ by culture).
    - Age_c_sq captures nonlinearity common across cultures (if desired, culture-specific quadratic terms could be added similarly).

    Returns the fitted statsmodels result object.
    """
    import statsmodels.formula.api as smf

    # Ensure the required transformed columns exist
    needed = ['MajorityChoice', 'Age_c', 'Age_c_sq', 'IsGirl', 'majority_first', 'culture']
    missing = [c for c in needed if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Build formula. Use categorical culture (C(culture)) and an interaction term Age_c:C(culture)
    formula = 'MajorityChoice ~ Age_c + Age_c_sq + IsGirl + majority_first + C(culture) + Age_c:C(culture)'

    # Fit logistic regression (Binomial family)
    model_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    # Optionally, one may compute robust SEs clustered by culture; here we return the fitted model object.
    return model_fit


