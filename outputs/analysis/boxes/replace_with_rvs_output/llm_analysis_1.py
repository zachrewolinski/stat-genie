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
    Transform the raw dataset into the dataframe required for modeling.

    Outputs (columns used in the model):
      - y: outcome (1,2,3) unchanged
      - age_c: centered age
      - culture_2..culture_8: dummy variables for culture (reference = culture 1)
      - age_c:culture_*: interaction terms between centered age and each culture dummy
      - is_boy: gender recoded (0 = girl, 1 = boy)
      - majority_first: kept as-is (0/1)
    """
    df = df.copy()

    # Required columns present check (will raise if missing)
    required = ['y', 'age', 'gender', 'culture', 'majority_first']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing values in key columns
    df = df.dropna(subset=required)

    # Ensure correct dtypes
    df['y'] = df['y'].astype(int)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce').astype(int)
    df['culture'] = pd.to_numeric(df['culture'], errors='coerce').astype(int)
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').astype(int)

    # Re-drop rows if conversion produced NaNs
    df = df.dropna(subset=['age', 'gender', 'culture', 'majority_first'])

    # Create centered age
    df['age_c'] = df['age'] - df['age'].mean()

    # Recode gender: 1=girl, 2=boy in schema -> create is_boy (0/1)
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # Create culture dummies, reference = culture 1
    culture_dummies = pd.get_dummies(df['culture'].astype(int), prefix='culture', drop_first=True)
    # Ensure dummy columns are named culture_2..culture_8 where applicable
    # (get_dummies will produce those names automatically because prefix + category)
    df = pd.concat([df.reset_index(drop=True), culture_dummies.reset_index(drop=True)], axis=1)

    # Identify culture dummy columns that were created
    culture_dummy_cols = [c for c in df.columns if c.startswith('culture_')]
    culture_dummy_cols = sorted(culture_dummy_cols, key=lambda x: int(x.split('_')[1]))

    # Create interactions between centered age and each culture dummy
    for col in culture_dummy_cols:
        inter_name = f'age_c:{col}'
        df[inter_name] = df['age_c'] * df[col]

    # Final housekeeping: keep only needed columns (but retain y)
    model_cols = ['y', 'age_c'] + culture_dummy_cols + [f'age_c:{c}' for c in culture_dummy_cols] + ['is_boy', 'majority_first']
    # Some cultures might not be present in a particular dataset subset; ensure only present columns are selected
    final_cols = [c for c in model_cols if c in df.columns]
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a multinomial logistic regression predicting categorical choice 'y' (1=undemonstrated, 2=majority, 3=minority)
    from age, culture dummies, their interactions, and controls (is_boy, majority_first).

    Returns the fitted statsmodels MNLogit result object. Prints the model summary and, if available, average marginal effects.
    """
    # Copy to avoid modifying caller's frame
    df = df.copy()

    # Identify predictors dynamically (based on transform output)
    culture_dummy_cols = [c for c in df.columns if c.startswith('culture_')]
    interaction_cols = [c for c in df.columns if c.startswith('age_c:culture_')]

    # Base predictors
    exog_cols = ['age_c'] + culture_dummy_cols + interaction_cols + ['is_boy', 'majority_first']
    # Keep only columns that exist in df (in case some culture dummies were absent)
    exog_cols = [c for c in exog_cols if c in df.columns]

    # Add constant
    exog = sm.add_constant(df[exog_cols], has_constant='add')
    endog = df['y']

    # Fit multinomial logit
    model = sm.MNLogit(endog, exog)
    try:
        results = model.fit(maxiter=200, disp=False)
    except Exception as e:
        # Retry with a different solver if convergence problem
        results = model.fit(method='bfgs', maxiter=400, disp=False)

    # Print summary
    print(results.summary())

    # Compute and print (optional) average marginal effects for the baseline specification if available
    try:
        marg = results.get_margeff()
        print('\nAverage marginal effects:')
        print(marg.summary())
    except Exception:
        # not critical; continue
        pass

    return results


