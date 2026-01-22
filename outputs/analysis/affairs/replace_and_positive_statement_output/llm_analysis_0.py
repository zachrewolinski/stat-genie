from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_and_positive_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair affairs dataframe into an analysis-ready dataframe.
    Creates:
      - Children : binary 1/0 for children in marriage (from 'children')
      - Female   : binary 1/0 for gender (from 'gender')
      - AnyAffair : binary indicator (1 if affairs > 0)
    Ensures required controls are numeric and drops rows with missing values in variables used in models.
    Returns transformed dataframe with all columns used in modeling.
    """
    df = df.copy()

    # Standardize column names if needed (assume input uses lowercase names as schema)
    # Map children to binary Children (1=yes, 0=no)
    if 'children' in df.columns:
        df['Children'] = df['children'].map({
            'yes': 1,
            'no': 0,
            'Yes': 1,
            'No': 0
        })
    else:
        df['Children'] = np.nan

    # Map gender to Female binary (1 female, 0 male)
    if 'gender' in df.columns:
        df['Female'] = df['gender'].map({
            'female': 1,
            'male': 0,
            'Female': 1,
            'Male': 0
        })
    else:
        df['Female'] = np.nan

    # Create AnyAffair binary indicator
    if 'affairs' in df.columns:
        # Ensure numeric
        df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
        df['AnyAffair'] = (df['affairs'] > 0).astype(int)
    else:
        df['AnyAffair'] = np.nan

    # Ensure control columns exist and are numeric
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # Drop rows with missing values for variables we will use in the models
    required_for_model = ['affairs', 'Children', 'Female'] + numeric_cols
    df = df.dropna(subset=required_for_model)

    # Final safety cast to integer for binary indicators
    df['Children'] = df['Children'].astype(int)
    df['Female'] = df['Female'].astype(int)
    df['AnyAffair'] = df['AnyAffair'].astype(int)

    # Return only useful columns (keeps originals as well, but ensures needed columns present)
    keep_cols = ['affairs', 'AnyAffair', 'Children', 'Female'] + numeric_cols
    # If some extra columns exist in df this will keep them too, but we return at least keep_cols
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to assess whether having children decreases engagement in extramarital affairs:
      1) Negative binomial regression for count of affairs (accounts for overdispersion and many zeros).
      2) Logistic regression for the probability of having any affair (binary outcome: AnyAffair).

    Both models control for gender, age, years married, religiousness, education, occupation, and marriage rating.
    Returns a dictionary with fitted model results objects: {'nb_model': ..., 'logit_model': ...}.
    """
    # Required imports (statsmodels already imported as sm in the environment)
    import statsmodels.api as sm

    # Define independent variables (controls + treatment)
    model_vars = ['Children', 'Female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    X = df[model_vars]
    X = sm.add_constant(X, has_constant='add')

    # Dependent variables
    y_count = df['affairs']
    y_binary = df['AnyAffair']

    results = {}

    # 1) Negative Binomial for counts
    try:
        # Use GLM NegativeBinomial (allows variance > mean). Use robust covariance after fitting.
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial()).fit()
        # Attach robust-covariance version as well
        nb_robust = nb_model.get_robustcov_results(cov_type='HC3')
        results['nb_model'] = nb_model
        results['nb_model_robust'] = nb_robust
    except Exception as e:
        results['nb_error'] = str(e)

    # 2) Logistic regression for any affair
    try:
        logit = sm.Logit(y_binary, X).fit(disp=False)
        logit_robust = logit.get_robustcov_results(cov_type='HC3')
        results['logit_model'] = logit
        results['logit_model_robust'] = logit_robust
    except Exception as e:
        results['logit_error'] = str(e)

    # Print brief summaries to console (optional) and return results dict
    # NB: consumers of this function can inspect .summary() of returned models
    if 'nb_model' in results:
        print('Negative Binomial (coefficients):')
        print(results['nb_model_robust'].summary())
    if 'logit_model' in results:
        print('\nLogistic regression for AnyAffair (coefficients):')
        print(results['logit_model_robust'].summary())

    return results


