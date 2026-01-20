from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep only rows with the variables needed for analysis
    required_cols = ['y', 'age', 'culture', 'gender', 'majority_first']
    df = df.dropna(subset=required_cols)

    # Dependent variable: create binary indicator for choosing majority (y == 2)
    # According to schema: y: 1=unchosen option, 2=majority option, 3=minority option
    df['is_majority'] = (df['y'] == 2).astype(int)

    # Independent variable: center age and add quadratic term to capture non-linear development
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c2'] = df['age_c'] ** 2

    # Culture: ensure categorical type for formula-based modeling
    # Keep column name 'culture' (used in the model as C(culture))
    df['culture'] = df['culture'].astype('category')

    # Gender: recode to binary 0/1 (0 = girl (original 1), 1 = boy (original 2))
    # If other codings appear, map conservatively: treat 1 as girl, 2 as boy where present
    df['gender_bin'] = df['gender'].map({1: 0, 2: 1})
    # If mapping produced NaNs because of unexpected codes, fill with original values coerced to 0/1
    if df['gender_bin'].isnull().any():
        # fallback: treat any non-1 code as 1
        df['gender_bin'] = df['gender_bin'].fillna((df['gender'] != 1).astype(int))

    # majority_first should already be 0/1; coerce to int
    df['majority_first'] = df['majority_first'].astype(int)

    # Final check: drop rows with any remaining NA in model columns
    model_cols = ['is_majority', 'age_c', 'age_c2', 'culture', 'gender_bin', 'majority_first']
    df = df.dropna(subset=model_cols)

    # Return transformed dataframe containing all columns required for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf

    # Fit a logistic regression predicting probability of choosing the majority option.
    # We model linear and quadratic age terms and allow culture to moderate both terms by including interactions.
    # Controls: gender_bin, majority_first.
    # Formula explanation:
    #   is_majority ~ (age_c + age_c2) * C(culture) + gender_bin + majority_first
    # expands to main effects of age_c, age_c2, culture, interactions age_c:culture and age_c2:culture,
    # plus controls.

    formula = 'is_majority ~ (age_c + age_c2) * C(culture) + gender_bin + majority_first'

    # Fit logistic regression (maximum likelihood)
    # Use disp=False to suppress convergence output; raise exceptions if convergence fails.
    model_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    # Return the fitted model object (call .summary() outside if you want textual summary)
    return model_fit


