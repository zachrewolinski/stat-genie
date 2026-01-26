from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/positive_leading_statement_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling-ready dataframe.

    Outputs (columns created and kept):
      - social_choice: binary (1 if child chose a demonstrated option: majority or minority; 0 if chose undemonstrated option)
      - majority_choice: binary (1 if y == 2, i.e., chose majority option)
      - minority_choice: binary (1 if y == 3, i.e., chose minority option)
      - age_centered: age minus mean(age)
      - age_centered2: squared centered age
      - gender_male: 1 if gender == 2 (boy), 0 if gender == 1 (girl)
      - majority_first: kept as-is (0/1)
      - culture: original culture id (kept)
      - culture_* : dummy columns for cultures 2..8 (culture_1 is baseline)
      - age_x_culture_* : interaction columns between age_centered and each culture dummy

    The function drops rows missing any of the core variables (y, age, culture, gender, majority_first).
    """

    # Work on a copy
    df = df.copy()

    # Drop rows with missing critical values
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Dependent variables: binary indicators
    df['majority_choice'] = (df['y'] == 2).astype(int)
    df['minority_choice'] = (df['y'] == 3).astype(int)
    df['social_choice'] = df['y'].isin([2, 3]).astype(int)

    # Age transformations: center and quadratic term
    df['age_centered'] = df['age'] - df['age'].mean()
    df['age_centered2'] = df['age_centered'] ** 2

    # Control: gender -> male indicator (1 = boy, 0 = girl)
    df['gender_male'] = (df['gender'] == 2).astype(int)

    # Ensure majority_first is numeric 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    # Culture: keep original and build dummies (drop_first=True -> baseline culture_1)
    df['culture'] = df['culture'].astype(int)
    culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)
    # Ensure full set of expected dummy columns exist (culture_2..culture_8). If any are missing (no observations), create them with zeros.
    expected = [f'culture_{i}' for i in range(2, 9)]
    for col in expected:
        if col not in culture_dummies.columns:
            culture_dummies[col] = 0
    # Reorder columns to a consistent order
    culture_dummies = culture_dummies[expected]

    df = pd.concat([df.reset_index(drop=True), culture_dummies.reset_index(drop=True)], axis=1)

    # Interaction terms: age_centered x each culture dummy
    interaction_cols = []
    for col in expected:
        inter_col = f'age_x_{col}'
        df[inter_col] = df['age_centered'] * df[col]
        interaction_cols.append(inter_col)

    # Final column list to keep (useful for downstream modeling)
    keep_cols = [
        'y', 'culture',
        'social_choice', 'majority_choice', 'minority_choice',
        'age_centered', 'age_centered2',
        'gender_male', 'majority_first'
    ] + expected + interaction_cols

    # Return only the columns we will use for modeling and inspection
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit logistic regression models to test whether (a) children's reliance on social information (social_choice)
    and (b) preference for the majority (majority_choice) and (c) preference for the minority (minority_choice)
    vary by age (developmental stage) and culture.

    Modeling approach:
      - Three separate binomial logistic regressions (Logit): social_choice, majority_choice, minority_choice.
      - Predictors: age_centered (linear), age_centered2 (quadratic), culture dummies (culture_2..culture_8),
        interactions age_centered x culture_dummies, and controls gender_male and majority_first.
      - Interactions test whether the developmental trajectory of choosing social options differs across cultures.

    Returns a dict with fitted statsmodels result objects and prints summaries.
    """

    # Work on copy
    df = df.copy()

    # Identify culture dummy columns (expected names)
    culture_dummy_cols = [f'culture_{i}' for i in range(2, 9) if f'culture_{i}' in df.columns]
    interaction_cols = [f'age_x_{col}' for col in culture_dummy_cols]

    base_predictors = ['age_centered', 'age_centered2', 'gender_male', 'majority_first']
    predictors = base_predictors + culture_dummy_cols + interaction_cols

    # Ensure predictors exist in df
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        raise ValueError(f"Missing predictor columns from transformed dataframe: {missing}")

    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')

    results = {}

    # Helper to fit and store model
    def fit_logit(y_col):
        y = df[y_col].astype(int)
        model = sm.Logit(y, X)
        try:
            res = model.fit(disp=False, method='lbfgs')
        except Exception:
            # Fall back to default solver if lbfgs fails
            res = model.fit(disp=False)
        return res

    # Fit models
    results['social_model'] = fit_logit('social_choice')
    results['majority_model'] = fit_logit('majority_choice')
    results['minority_model'] = fit_logit('minority_choice')

    # Print concise summaries for quick inspection
    print('\n=== Social choice model (chose demonstrated option vs unchosen) ===')
    print(results['social_model'].summary())

    print('\n=== Majority-choice model (chose majority vs others) ===')
    print(results['majority_model'].summary())

    print('\n=== Minority-choice model (chose minority vs others) ===')
    print(results['minority_model'].summary())

    return results


