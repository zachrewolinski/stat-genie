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
    Transform the raw dataset into a dataframe suitable for modeling.

    Outputs (columns created/kept):
      - Choice: integer copy of original y (1=unchosen,2=majority,3=minority)
      - Choice_code: zero-based code (0,1,2) for statsmodels MNLogit
      - MajorityChoice: binary indicator 1 if child chose majority (y==2)
      - IsBoy: binary gender indicator (1 if gender==2 (boy), 0 if gender==1 (girl))
      - age_c: centered age
      - age_c2: squared centered age
      - AgeGroup: binned age group for descriptive work ('4-6','7-9','10-14')
      - culture_2..culture_8: dummy variables for cultures 2..8 (culture_1 is baseline)
      - age_c:culture_#: interaction terms between centered age and each culture dummy
      - majority_first: cast to int
    """
    df = df.copy()

    # Drop rows with missing critical variables
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Dependent variables
    df['Choice'] = df['y'].astype(int)
    # zero-based code for MNLogit (0,1,2)
    df['Choice_code'] = df['Choice'] - 1
    # binary outcome: did the child pick the majority demonstration?
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Controls
    # gender: original coding 1=girl, 2=boy -> IsBoy binary
    df['IsBoy'] = (df['gender'].astype(int) == 2).astype(int)

    # Center age and add quadratic term for potential nonlinearity
    df['age'] = df['age'].astype(float)
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c2'] = df['age_c'] ** 2

    # Age group for descriptive checks (not required for the main models but useful)
    bins = [3.5, 6.5, 9.5, 14.5]
    labels = ['4-6', '7-9', '10-14']
    df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

    # Culture dummies: create explicit dummy columns for culture_2..culture_8 (culture_1 = baseline)
    # This ensures consistent column names even if some cultures are absent in a particular subset.
    culture_vals = df['culture'].astype(int)
    culture_dummies = pd.get_dummies(culture_vals, prefix='culture')

    desired_cult_cols = [f'culture_{i}' for i in range(2, 9)]
    # Add any missing dummy columns with zeros so subsequent code can always refer to these names
    for c in desired_cult_cols:
        if c not in culture_dummies.columns:
            culture_dummies[c] = 0
    # Keep only the desired columns (2..8). If some cultures weren't present, those columns are zeros.
    culture_df = culture_dummies.reindex(columns=desired_cult_cols, fill_value=0)
    df = pd.concat([df.reset_index(drop=True), culture_df.reset_index(drop=True)], axis=1)

    # Interaction terms: age_c multiplied by each culture dummy
    for c in desired_cult_cols:
        df[f'age_c:{c}'] = df['age_c'] * df[c]

    # Ensure majority_first is integer 0/1
    df['majority_first'] = df['majority_first'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run two complementary models to answer the research question:
      1) A multinomial logistic regression predicting Choice (unchosen / majority / minority) from age, culture, their interaction, and controls.
      2) A binary logistic regression predicting whether the child chose the majority option (MajorityChoice) with the same predictors. This gives a focused test of majority preference.

    Returns a dictionary containing the fitted statsmodels results objects for both models.
    """
    # Ensure we operate on a copy
    data = df.copy()

    # Define the exogenous columns to include. Only keep those that exist in data (robust to missing cultures).
    main_cols = ['age_c', 'age_c2', 'IsBoy', 'majority_first']
    culture_cols = [f'culture_{i}' for i in range(2, 9)]
    interaction_cols = [f'age_c:culture_{i}' for i in range(2, 9)]

    exog_cols = main_cols + culture_cols + interaction_cols
    exog_cols = [c for c in exog_cols if c in data.columns]

    # Prepare X and add intercept
    X = data[exog_cols]
    X = sm.add_constant(X, has_constant='add')

    results = {}

    # 1) Multinomial logistic regression: Choice_code should be 0,1,2
    if 'Choice_code' not in data.columns:
        raise ValueError("Choice_code column not found in dataframe. Run transform() first.")

    endog = data['Choice_code']
    # Fit MNLogit (note: statsmodels' MNLogit uses a reference category implicitly)
    mnlogit_model = sm.MNLogit(endog, X)
    mnlogit_res = mnlogit_model.fit(method='newton', maxiter=100, disp=False)
    results['mnlogit'] = mnlogit_res

    # 2) Binary logistic regression for choosing the majority option
    if 'MajorityChoice' not in data.columns:
        raise ValueError("MajorityChoice column not found in dataframe. Run transform() first.")

    y_bin = data['MajorityChoice']
    logit_model = sm.Logit(y_bin, X)
    logit_res = logit_model.fit(disp=False)
    results['logit_majority'] = logit_res

    # Return fitted model objects (caller can call .summary() on each)
    return results


