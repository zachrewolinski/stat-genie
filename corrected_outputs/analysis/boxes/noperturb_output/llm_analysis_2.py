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
    # Ensure required raw columns exist
    required = ['y', 'age', 'culture', 'gender', 'majority_first']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input df: {missing}")

    # Drop rows with missing values in the key variables
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first']).copy()

    # Dependent variable: majority choice (y==2 means chose majority)
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Center age for interpretability of interactions
    df['age_c'] = df['age'] - df['age'].mean()

    # Normalize/standardize gender into a binary indicator: 1 = boy, 0 = girl
    # Original coding: 1 = girl, 2 = boy
    df['gender_boy'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is numeric binary (0/1)
    df['majority_first'] = df['majority_first'].astype(int)

    # Culture: treat as categorical with explicit categories 1..8 to produce a consistent set of dummy columns
    expected_cultures = list(range(1, 9))
    df['culture'] = pd.Categorical(df['culture'].astype(int), categories=expected_cultures)

    # Create dummy variables for cultures with site 1 as the reference (drop_first=True => drops culture_1)
    dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)

    # Ensure all expected dummy columns exist (culture_2 .. culture_8)
    culture_columns = [f'culture_{i}' for i in range(2, 9)]
    for col in culture_columns:
        if col in dummies.columns:
            df[col] = dummies[col].astype(int)
        else:
            # If a culture is missing in the sample, create a zero column with the original index
            df[col] = 0

    # Create interaction terms: age_c x each culture dummy
    interaction_columns = []
    for col in culture_columns:
        inter_col = 'age_x_' + col
        df[inter_col] = df['age_c'] * df[col]
        interaction_columns.append(inter_col)

    # Final columns of interest (kept in dataframe for modeling)
    # MajorityChoice, age_c, gender_boy, majority_first, culture_2..culture_8, age_x_culture_2..age_x_culture_8

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Specify columns used in the logistic regression
    culture_cols = [f'culture_{i}' for i in range(2, 9)]
    interaction_cols = [f'age_x_culture_{i}' for i in range(2, 9)]
    base_cols = ['age_c', 'gender_boy', 'majority_first']
    X_cols = base_cols + culture_cols + interaction_cols

    # Check that all required columns exist
    missing = [c for c in X_cols + ['MajorityChoice'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['MajorityChoice'].astype(int)

    # Fit a logistic regression (binomial) predicting choosing the majority option
    # We model interactions between centered age and culture dummies to capture how age-related change differs across sites.
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Return the fitted results object (results.summary() can be printed by the caller)
    return results


