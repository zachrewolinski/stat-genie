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
    Transform the raw dataset into the analysis dataframe.

    Produces the following columns (kept/created):
    - majority_choice: binary DV (1 if y==2 (majority), else 0)
    - age_c: age mean-centered (continuous IV)
    - culture: categorical site identifier (categorical IV)
    - is_male: binary control (1 = boy, 0 = girl)
    - majority_first: binary control (0/1), coerced to integer

    The function drops rows with missing values in the variables needed for modeling.
    """
    # work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing core variables
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Dependent variable: majority_choice (1 if chose majority option (y == 2), else 0)
    df['majority_choice'] = (df['y'] == 2).astype(int)

    # Age: ensure numeric and mean-center
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    # drop rows that became NaN after coercion
    df = df.dropna(subset=['age'])
    df['age_c'] = df['age'] - df['age'].mean()

    # Culture: treat as categorical factor (keep original codes but set dtype to category)
    # This column is used as C(culture) in the model formula
    df['culture'] = df['culture'].astype('category')

    # Gender -> is_male (1 = boy (gender==2), 0 = girl (gender==1))
    df['is_male'] = (pd.to_numeric(df['gender'], errors='coerce') == 2).astype(int)

    # majority_first: ensure numeric 0/1
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').astype(int)

    # Final drop in case any coercion produced NaNs
    df = df.dropna(subset=['majority_choice', 'age_c', 'culture', 'is_male', 'majority_first'])

    # Return dataframe containing all columns (including originals); required analysis columns are present
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting the probability of choosing the majority option.

    Model formula:
      majority_choice ~ age_c * C(culture) + is_male + majority_first

    Interpretation: main effect of age (developmental trend), main effects of culture (differences in baseline propensity
    to follow majority), and age-by-culture interactions (different developmental trajectories across cultures). Controls
    for child's gender and whether the majority was demonstrated first.

    Returns the fitted statsmodels results object (LogitResults) so the caller can inspect summary, coefficients, CIs, etc.
    """
    import statsmodels.formula.api as smf

    # Ensure the culture column is categorical (transform() should have done this)
    if not pd.api.types.is_categorical_dtype(df['culture']):
        df['culture'] = df['culture'].astype('category')

    formula = 'majority_choice ~ age_c * C(culture) + is_male + majority_first'

    # Fit logistic regression (binomial logit)
    # Use disp=False to avoid printing during fit
    model = smf.logit(formula=formula, data=df).fit(disp=False)

    # Return the fitted model object. The caller can call model.summary(), model.params, model.conf_int(), etc.
    return model


