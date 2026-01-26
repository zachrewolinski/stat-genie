from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all required data cleaning and feature engineering.

    Inputs:
    - df: original dataframe with columns described in the schema (y, gender, age, majority_first, culture, ...)

    Outputs (adds/returns columns used in the model):
    - Choice: categorical labels ('unchosen','majority','minority')
    - Choice_code: numeric codes 0,1,2 (0=unchosen,1=majority,2=minority)
    - gender_b: 0/1 (0=girl,1=boy)
    - age_c: centered age
    - age_c2: squared centered age
    - culture_2..culture_8: dummy variables for culture (culture_1 used as reference and dropped)
    - age_c:culture_*: interaction terms between centered age and each culture dummy (excluding reference)

    The function will drop rows with missing values in the variables needed for the modeling.
    """
    df = df.copy()

    # Keep only rows with non-missing outcome and core predictors
    df = df.dropna(subset=['y', 'age', 'culture'])

    # Recode dependent variable y to labeled and numeric codes
    # According to schema: 1 = unchosen option, 2 = majority option, 3 = minority option
    mapping = {1: 'unchosen', 2: 'majority', 3: 'minority'}
    df['Choice'] = df['y'].map(mapping)
    # Force categorical with known ordering (important for reproducibility)
    df['Choice'] = pd.Categorical(df['Choice'], categories=['unchosen', 'majority', 'minority'])
    df['Choice_code'] = df['Choice'].cat.codes  # 0,1,2

    # Recode gender to binary (control)
    # Schema: 1 = girl, 2 = boy
    df['gender_b'] = df['gender'].map({1: 0, 2: 1})

    # Ensure majority_first is binary (0/1)
    df['majority_first'] = df['majority_first'].astype(float).fillna(0).astype(int)

    # Center age and add quadratic term
    df['age_c'] = df['age'] - df['age'].mean()
    df['age_c2'] = df['age_c'] ** 2

    # Create culture dummies. We will create dummies for observed culture values and set culture_1 as reference
    # (drop the reference from the exogenous matrix to avoid multicollinearity).
    df['culture'] = df['culture'].astype(int)
    culture_dummies = pd.get_dummies(df['culture'], prefix='culture')

    # Ensure dummies for expected culture levels 1..8 exist (if a level is missing in the sample, the column will not exist)
    # but we will create columns for culture_2..culture_8 if they are missing and fill with zeros so the model code can always refer to them.
    expected = [f'culture_{i}' for i in range(1, 9)]
    for col in expected:
        if col not in culture_dummies.columns:
            culture_dummies[col] = 0

    # Attach dummies to df
    df = pd.concat([df, culture_dummies], axis=1)

    # Choose reference category culture_1 (keep the column but we will not include it in the model matrix).
    # Define the list of culture dummy columns to include as IVs (exclude reference 'culture_1')
    culture_dummy_cols = [col for col in expected if col != 'culture_1']

    # Create interaction terms between centered age and each culture dummy (excluding reference)
    for col in culture_dummy_cols:
        interaction_name = f'age_c:{col}'
        df[interaction_name] = df['age_c'] * df[col]

    # Final list of columns that will be used as exogenous variables in the model (no constant here)
    # The model function will add a constant.
    df['__exog_cols__'] = ','.join(['age_c', 'age_c2', 'gender_b', 'majority_first'] + culture_dummy_cols + [f'age_c:{c}' for c in culture_dummy_cols])

    # Return the dataframe with new columns. Downstream model() will read the columns listed above.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a multinomial logistic regression predicting Choice_code (0=unchosen,1=majority,2=minority)
    from age (linear & quadratic), culture dummies, their interactions (age x culture), and controls (gender, majority_first).

    Returns a dictionary with keys:
    - 'model_raw': the fitted MNLogitResults object
    - 'model_robust': robust covariance adjusted results clustered by culture
    - 'margeff': marginal effects summary (as text)
    - 'exog_cols': list of exogenous columns used (for reproducibility)
    """
    import statsmodels.api as sm

    df = df.copy()

    # Reconstruct exogenous columns list (matches what transform() stored in '__exog_cols__')
    if '__exog_cols__' in df.columns:
        exog_cols = df['__exog_cols__'].dropna().iloc[0].split(',')
    else:
        # Fallback: assemble according to convention
        exog_cols = ['age_c', 'age_c2', 'gender_b', 'majority_first']
        exog_cols += [col for col in df.columns if col.startswith('culture_') and col != 'culture_1']
        exog_cols += [col for col in df.columns if col.startswith('age_c:culture_')]

    # Ensure exog columns exist in df (if some culture levels were absent they will be zeros)
    for col in exog_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[exog_cols].astype(float)
    X = sm.add_constant(X, prepend=True)

    y = df['Choice_code'].astype(int)

    # Fit multinomial logit
    # Note: statsmodels' MNLogit models J-1 sets of coefficients for J outcome categories.
    mnlogit = sm.MNLogit(y, X)
    try:
        res = mnlogit.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        # fallback to default solver if newton fails
        res = mnlogit.fit(method='bfgs', maxiter=200, disp=False)

    # Compute cluster-robust covariance (clustered by culture site)
    # If culture column exists, use it; otherwise fallback to no clustering
    if 'culture' in df.columns:
        try:
            robust_res = res.get_robustcov_results(cov_type='cluster', groups=df['culture'].values)
        except Exception:
            # if clustering fails, just return the original results as robust_res
            robust_res = res
    else:
        robust_res = res

    # Marginal effects (average marginal effects)
    try:
        mfx = res.get_margeff(at='overall')
        mfx_summary = mfx.summary().as_text()
    except Exception as e:
        mfx_summary = f"Could not compute marginal effects: {e}"

    # Return results and some diagnostic output
    out = {
        'model_raw': res,
        'model_robust': robust_res,
        'margeff': mfx_summary,
        'exog_cols': X.columns.tolist()
    }
    return out


