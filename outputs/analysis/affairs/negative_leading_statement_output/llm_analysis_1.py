from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/negative_leading_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (1978) affairs dataset into an analysis-ready dataframe.

    Produces the following additional columns (kept for modeling):
      - Children: binary (1 if 'children' == 'yes', 0 if 'no')
      - Female: binary (1 if gender == 'female', 0 otherwise)
      - AnyAffair: binary (1 if affairs > 0, 0 if affairs == 0)
      - LogAffairs1: continuous log(affairs + 1) for OLS robustness

    The function also drops rows with missing values in columns required for modeling.
    """
    df = df.copy()

    # Normalize string columns to lower-case where necessary and handle missing values
    if 'children' in df.columns:
        df['children'] = df['children'].astype(str).str.strip().str.lower()
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.lower()

    # Create binary Children column: 1 if yes, 0 if no
    df['Children'] = df['children'].map({'yes': 1, 'no': 0})

    # Create Female indicator
    df['Female'] = df['gender'].map({'female': 1, 'male': 0})

    # Create AnyAffair binary
    df['AnyAffair'] = (df['affairs'].fillna(0) > 0).astype(int)

    # Create log(affairs + 1) for OLS robustness
    # Ensure numeric type
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
    df['LogAffairs1'] = np.log1p(df['affairs'].fillna(0))

    # Select and require the columns needed for modeling
    required_cols = [
        'affairs', 'Children', 'AnyAffair', 'LogAffairs1',
        'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'Female'
    ]

    # If any required column not present, raise a clear error
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing values in required modeling columns
    df = df.dropna(subset=required_cols)

    # Convert numeric columns to proper dtype
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'LogAffairs1']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop again any rows that became NaN after conversion
    df = df.dropna(subset=numeric_cols + ['Children', 'Female', 'AnyAffair'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit three complementary models to assess whether having children decreases engagement in extramarital affairs.

    Models fit:
      1) Logistic regression (Logit) predicting AnyAffair (binary: any affair in past year) -- interpretable as odds of any affair.
      2) Negative binomial regression (GLM) predicting affairs (count) -- suitable for overdispersed counts.
      3) OLS on LogAffairs1 = log(affairs + 1) as a robustness check (heteroskedasticity-robust SEs).

    All models include the same set of controls: age, yearsmarried, religiousness, education, occupation, rating, Female.

    Returns a dict with fitted results objects and printed summaries.
    """
    results = {}
    # Ensure df is the transformed dataframe from transform()
    df = df.copy()

    # Define covariates
    control_vars = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'Female']
    exog_vars = ['Children'] + control_vars

    # Design matrices with constant
    X = df[exog_vars]
    X = sm.add_constant(X, has_constant='add')

    # Dependent variables
    y_count = df['affairs']
    y_bin = df['AnyAffair']
    y_log1p = df['LogAffairs1']

    # 1) Logistic regression for probability of any affair
    try:
        logit_model = sm.Logit(y_bin, X)
        logit_res = logit_model.fit(disp=False)
        # Robust (HC3) standard errors for inference
        logit_res_robust = logit_res.get_robustcov_results(cov_type='HC3')
        results['logit'] = logit_res_robust
        print('\nLogistic regression (AnyAffair) results:')
        print(logit_res_robust.summary())
    except Exception as e:
        results['logit_error'] = str(e)
        print('Logit model failed:', e)

    # 2) Negative binomial regression for count of affairs
    try:
        # Use GLM NegativeBinomial family (log link) as a flexible count model
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
        results['negbin'] = nb_res
        print('\nNegative binomial (count) regression results:')
        print(nb_res.summary())
    except Exception as e:
        results['negbin_error'] = str(e)
        print('Negative binomial model failed:', e)

    # 3) OLS on log(affairs + 1) as a robustness check
    try:
        ols_model = sm.OLS(y_log1p, X).fit(cov_type='HC3')
        results['ols_log1p'] = ols_model
        print('\nOLS on log(affairs+1) results (HC3 SEs):')
        print(ols_model.summary())
    except Exception as e:
        results['ols_error'] = str(e)
        print('OLS model failed:', e)

    # For interpretability, compute and print the marginal effect (approx) of Children from the negative binomial (percent change)
    try:
        if 'negbin' in results and not isinstance(results['negbin'], str):
            coef_children = results['negbin'].params.get('Children', np.nan)
            pct_change = (np.exp(coef_children) - 1) * 100
            print(f"\nApprox. percent change in expected affairs associated with Children=1 (Ne gBin): {pct_change:.2f}%")
            results['negbin_children_pct_change'] = pct_change
    except Exception:
        pass

    return results


