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

    # Drop rows with missing critical variables
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Dependent variable: binary indicator for choosing the majority option (y==2)
    df['ChoseMajority'] = (df['y'] == 2).astype(int)

    # Center age for interpretability and include a quadratic term to capture non-linearity
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c2'] = df['age_c'] ** 2

    # Gender: create female indicator (1 = girl, 0 = boy)
    df['gender_female'] = (df['gender'] == 1).astype(int)

    # Ensure majority_first is binary (0/1)
    df['majority_first'] = df['majority_first'].astype(int)

    # Culture: convert to a categorical label (e.g., 'C1', 'C2', ...). Keep original numeric codes in case needed.
    # This column will be used to create dummy variables in the model.
    df['culture_cat'] = 'C' + df['culture'].astype(int).astype(str)

    # Final dataframe contains all columns needed for modeling
    # (also keep original columns so users can inspect raw values)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    df = df.copy()

    # Response variable
    y = df['ChoseMajority']

    # Create culture dummy variables (drop_first to avoid perfect multicollinearity)
    culture_dummies = pd.get_dummies(df['culture_cat'], prefix='culture', drop_first=True)

    # Base predictors: age (centered), quadratic age, gender, order effect
    X_base = df[['age_c', 'age_c2', 'gender_female', 'majority_first']]

    # Combine base predictors and culture dummies
    X = pd.concat([X_base, culture_dummies], axis=1)

    # Add interactions between age_c and each culture dummy to allow culture-specific age slopes
    for col in culture_dummies.columns:
        X[f'{col}_x_age'] = X[col] * X['age_c']

    # Add constant
    X = sm.add_constant(X)

    # Fit a logistic regression (binomial GLM) with culture fixed effects and age-by-culture interactions
    # We use statsmodels Logit for a straightforward maximum-likelihood binary model
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Return the fitted results object for downstream inspection (coefficients, std errors, summary, predictions)
    return results


