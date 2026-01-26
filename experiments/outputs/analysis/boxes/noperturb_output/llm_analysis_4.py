from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/noperturb_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep rows with required information
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Recode dependent variable to 0..2 for multinomial modeling
    # Original coding: 1 = unchosen option, 2 = majority option, 3 = minority option
    df['y_cat'] = df['y'].map({1: 0, 2: 1, 3: 2}).astype(int)

    # Age: center to improve interpretability and numerical stability
    df['Age_c'] = df['age'].astype(float) - df['age'].astype(float).mean()
    df['Age_c_sq'] = df['Age_c'] ** 2

    # Gender: create binary indicator for male (original: 1=girl, 2=boy)
    df['gender_male'] = df['gender'].apply(lambda x: 1 if x == 2 else 0).astype(int)

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Culture: keep as integer categorical identifier (1..8)
    df['culture'] = df['culture'].astype(int)

    # Final sanity: drop rows with any NA in the model columns
    model_cols = ['y_cat', 'Age_c', 'Age_c_sq', 'gender_male', 'majority_first', 'culture']
    df = df.dropna(subset=model_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits two complementary models to answer the research question:
      1) A multinomial logistic regression predicting the 3-choice outcome (unchosen / majority / minority)
         from age (linear + quadratic), culture (categorical), their interaction (Age x Culture),
         and controls (gender, majority_first). This tests whether reliance on social information (choice distribution)
         differs with age and across cultures and whether developmental trajectories vary by culture.

      2) A binary logistic regression predicting whether the child chose the majority option (majority vs. not)
         using the same predictors. This directly targets "preference for majority cues".

    Returns a dict with the fitted results objects.
    """

    # Required imports inside the model scope
    import statsmodels.api as sm
    import pandas as pd

    # Build exogenous (X) matrix
    base_cols = ['Age_c', 'Age_c_sq', 'gender_male', 'majority_first']
    X_base = df[base_cols].copy()

    # Create culture dummies (drop_first to use one site as reference)
    culture_dummies = pd.get_dummies(df['culture'].astype(str), prefix='culture', drop_first=True)

    # Combine base predictors and culture dummies
    X = pd.concat([X_base, culture_dummies], axis=1)

    # Add Age x Culture interaction terms to allow culture to moderate developmental trajectories
    for col in culture_dummies.columns:
        X[f'{col}_x_Age_c'] = X[col] * X['Age_c']

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Endogenous (y): multinomial categories coded as integers 0..2
    y = df['y_cat'].astype(int)

    # 1) Multinomial logistic regression (choice among 3 options)
    mnlogit_mod = sm.MNLogit(y, X)
    try:
        mnlogit_res = mnlogit_mod.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        # fallback to default fit (may print warnings)
        mnlogit_res = mnlogit_mod.fit(disp=False)

    # 2) Binary logistic regression for majority preference (1 = majority chosen, 0 = otherwise)
    y_majority = (y == 1).astype(int)
    logit_mod = sm.Logit(y_majority, X)
    try:
        logit_res = logit_mod.fit(disp=False)
    except Exception:
        logit_res = logit_mod.fit(disp=False, method='newton', maxiter=100)

    # Return the fitted results objects; the caller can inspect .summary() or other attributes
    return {
        'mnlogit_result': mnlogit_res,
        'logit_majority_result': logit_res,
        'exog_columns': list(X.columns)
    }


