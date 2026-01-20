from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/noperturb_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe into the analysis-ready dataframe.

    Produced columns (used by the model):
    - y_mn: integer coded 0/1/2 for multinomial outcome (0=unchosen, 1=majority, 2=minority)
    - Age_c: age (years) mean-centered
    - Age_c2: squared centered age
    - gender_female: binary indicator (1 if girl, 0 if boy)
    - majority_first: binary indicator (0/1) (kept/coerced)
    - culture: categorical site ID (dtype=category)
    """
    df = df.copy()

    # Keep rows with required variables
    required_cols = ['y', 'age', 'gender', 'majority_first', 'culture']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')
    df['culture'] = df['culture'].astype('category')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['y', 'age', 'gender', 'majority_first', 'culture'])

    # Dependent variable for multinomial model: convert to 0-indexed integers
    # Original coding: 1 = unchosen option, 2 = majority option, 3 = minority option
    df['y_mn'] = (df['y'].astype(int) - 1).astype(int)

    # Center age and add quadratic term to capture non-linear development
    df['Age_c'] = df['age'] - df['age'].mean()
    df['Age_c2'] = df['Age_c'] ** 2

    # Gender: make female indicator (1 = girl (gender==1), 0 = boy)
    df['gender_female'] = (df['gender'] == 1).astype(int)

    # Ensure majority_first is 0/1 integer
    df['majority_first'] = df['majority_first'].astype(int)

    # Final check: ensure y_mn values are within {0,1,2}
    df = df[df['y_mn'].isin([0, 1, 2])]

    # Keep only columns relevant for analysis (but preserve others if desired)
    # Return full df with added columns; model function will pick needed columns by name
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a multinomial logistic regression predicting choice (unchosen / majority / minority)
    from age (linear + quadratic), culture (categorical), their interaction (Age_c * culture),
    and controlling for gender and whether the majority was shown first.

    Model specification (formula):
      y_mn ~ Age_c * C(culture) + Age_c2 + gender_female + majority_first

    Returns the fitted statsmodels results object for the multinomial logit.
    """
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Ensure culture is categorical for formula interface
    df['culture'] = df['culture'].astype('category')

    # Formula: age-by-culture interaction tests whether age slopes differ by culture
    formula = 'y_mn ~ Age_c * C(culture) + Age_c2 + gender_female + majority_first'

    # Fit multinomial logistic regression (reference category will be the first category, here y_mn==0)
    # Use Newton method for stability
    model_instance = smf.mnlogit(formula, data=df)
    results = model_instance.fit(method='newton', maxiter=200, disp=False)

    # Print a brief summary to console; primary return is the results object
    try:
        print(results.summary())
    except Exception:
        # summary printing can fail in some environments; ignore if so
        pass

    return results


