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
    # Keep only rows with essential variables present
    df = df.copy()
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Dependent variable: categorical choice (1=unchosen, 2=majority, 3=minority)
    df['Choice'] = df['y'].astype(int)

    # Age: keep original and create a centered version for modeling
    df['Age'] = df['age'].astype(float)
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Gender control: create binary IsBoy (0 = girl (gender==1), 1 = boy (gender==2))
    # If gender is missing, drop those rows to keep modeling simple
    if 'gender' in df.columns:
        df = df.dropna(subset=['gender'])
        df['IsBoy'] = df['gender'].apply(lambda x: 1 if int(x) == 2 else 0).astype(int)
    else:
        # If gender column absent, create IsBoy as 0 (no info) but warn user
        df['IsBoy'] = 0

    # Majority-first control: ensure integer 0/1
    if 'majority_first' in df.columns:
        df = df.dropna(subset=['majority_first'])
        df['MajorityFirst'] = df['majority_first'].astype(int)
    else:
        df['MajorityFirst'] = 0

    # Culture: keep original and create dummy variables (drop first category as reference)
    # Convert culture to integer/string to make stable dummy names
    df['Culture'] = df['culture'].astype(int)
    culture_dummies = pd.get_dummies(df['Culture'].astype(str), prefix='Culture', drop_first=True)
    # Ensure consistent dummy column order: Culture_2 ... Culture_8 (if present)
    # Concatenate dummies to dataframe
    df = pd.concat([df, culture_dummies], axis=1)

    # Create age-by-culture interaction terms for each dummy to allow culture-specific age slopes
    interaction_cols = []
    for col in culture_dummies.columns:
        inter_name = f'Age_c_{col}'
        df[inter_name] = df['Age_c'] * df[col]
        interaction_cols.append(inter_name)

    # Final dataframe contains at least the following columns used in the model:
    # 'Choice', 'Age_c', culture dummies (e.g., 'Culture_2'..), interaction cols (e.g., 'Age_c_Culture_2'..), 'IsBoy', 'MajorityFirst'
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build model design matrix for multinomial logistic regression
    # Ensure transform() has been applied already
    df = df.copy()

    # Identify culture dummy columns (those starting with 'Culture_') and their interactions (starting with 'Age_c_Culture_')
    culture_dummy_cols = [c for c in df.columns if c.startswith('Culture_')]
    interaction_cols = [c for c in df.columns if c.startswith('Age_c_Culture_')]

    # Exogenous variables: centered age, culture dummies, age-by-culture interactions, and controls
    exog_cols = ['Age_c'] + culture_dummy_cols + interaction_cols + ['IsBoy', 'MajorityFirst']

    # Drop rows with any missing values in exog or endog
    model_df = df.dropna(subset=['Choice'] + exog_cols)

    # Prepare X and y
    X = model_df[exog_cols]
    X = sm.add_constant(X, has_constant='add')
    y = model_df['Choice'].astype(int)  # values expected 1,2,3

    # Fit multinomial logistic regression (MNLogit). The reference category will be the first value (lowest label) by default.
    # Here Choice values are 1,2,3. MNLogit models the log-odds of each non-reference category vs reference.
    mnlogit = sm.MNLogit(y, X)
    results = mnlogit.fit(method='newton', maxiter=200, disp=False)

    # Return the fitted results object. The caller can do results.summary() to inspect coefficients.
    return results


