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
    Transform the raw Fair (1978) affairs dataset into a cleaned dataframe ready for modeling.

    Produces the following columns used in modeling:
      - affairs: numeric outcome (keeps original coding: 0,1,2,3,7,12)
      - children_yes: binary 1 if children == 'yes', 0 if 'no'
      - gender_male: binary 1 if gender == 'male', 0 if 'female'
      - age, yearsmarried, religiousness, education, occupation, rating: numeric controls

    Drops rows with missing values in outcome, treatment (children), or control variables.
    """
    df = df.copy()

    # Ensure affairs numeric (coerce problematic values to NaN)
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Standardize children indicator to lower-case strings then map
    df['children'] = df['children'].astype(str).str.strip().str.lower()
    df['children_yes'] = df['children'].map({'yes': 1, 'no': 0})

    # Standardize gender and create binary
    df['gender'] = df['gender'].astype(str).str.strip().str.lower()
    df['gender_male'] = df['gender'].map({'male': 1, 'female': 0})

    # Ensure numeric controls; coerce non-numeric to NaN
    for col in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only rows with non-missing required columns
    required_cols = ['affairs', 'children_yes', 'gender_male', 'age', 'yearsmarried',
                     'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=required_cols)

    # Reset index to make downstream processing easier
    df = df.reset_index(drop=True)

    # Return transformed dataframe. Column names here are used directly in the model code.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit multiple specifications to estimate the effect of having children on reported extramarital affairs.

    Primary model: Zero-Inflated Negative Binomial (ZINB) with the same covariates in both the count and inflation equations.
    Additional models: Zero-Inflated Poisson (ZIP) as fallback, and OLS with robust standard errors for a simple benchmark.

    Returns a dictionary with fitted model results and a short numeric summary (means and coefficient/p-value for children variable).
    """
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings

    # Count-model classes
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson
    except Exception:
        # If statsmodels version lacks these, raise an informative error
        raise ImportError('statsmodels.discrete.count_model ZeroInflated classes required (update statsmodels).')

    results = {}

    # Define outcome and covariates
    endog = df['affairs'].astype(float)
    exog_vars = ['children_yes', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'gender_male']
    exog = sm.add_constant(df[exog_vars], prepend=True)

    # 1) OLS benchmark with robust (HC3) standard errors
    ols_mod = sm.OLS(endog, exog)
    ols_res = ols_mod.fit(cov_type='HC3')
    results['ols'] = ols_res

    # 2) Zero-Inflated Negative Binomial (primary preferred model for overdispersed counts with many zeros)
    # Use same exog for count and inflation parts.
    zinb_res = None
    zip_res = None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        try:
            zinb_mod = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog, p=1)
            zinb_res = zinb_mod.fit(maxiter=200, method='newton', disp=False)
            results['zinb'] = zinb_res
        except Exception as e:
            results['zinb_error'] = str(e)
            # fallback to ZIP
            try:
                zip_mod = ZeroInflatedPoisson(endog, exog, exog_infl=exog)
                zip_res = zip_mod.fit(maxiter=200, disp=False)
                results['zip'] = zip_res
            except Exception as e2:
                results['zip_error'] = str(e2)

    # Helper to extract the coefficient and p-value for the children_yes effect from a fitted model
    def extract_child_coef(fitted):
        if fitted is None:
            return (None, None)
        params = getattr(fitted, 'params', None)
        pvalues = getattr(fitted, 'pvalues', None)
        if params is None or pvalues is None:
            return (None, None)
        # Try several possible parameter name patterns that may appear in different models
        candidates = ['children_yes', 'inflate_children_yes', 'children_yes?' ]
        for name in params.index:
            if name == 'children_yes' or name.endswith('.children_yes') or 'children_yes' in name:
                return (float(params[name]), float(pvalues[name]))
        # If not found, return None
        return (None, None)

    # Extract coefficients from preferred model in order: ZINB -> ZIP -> OLS
    coef = None
    pval = None
    if 'zinb' in results:
        coef, pval = extract_child_coef(results['zinb'])
        preferred = 'zinb'
    elif 'zip' in results:
        coef, pval = extract_child_coef(results['zip'])
        preferred = 'zip'
    else:
        coef, pval = extract_child_coef(results['ols'])
        preferred = 'ols'

    # Descriptive means by children status
    means_table = df.groupby('children_yes')['affairs'].agg(['mean', 'median', 'count', 'std']).to_dict()

    # Summarize direction and statistical evidence
    if coef is None:
        interpretation = 'Could not extract children coefficient from fitted models.'
    else:
        direction = 'decrease' if coef < 0 else ('increase' if coef > 0 else 'no change')
        significance = 'statistically significant' if (pval is not None and pval < 0.05) else 'not statistically significant'
        interpretation = (f"Preferred model = {preferred}. Children coefficient (count eqn) = {coef:.4f}, p = {pval:.4f} -> "
                          f"Estimated effect is a {direction} in reported affairs and is {significance} at the 5% level.")

    results['summary'] = {
        'preferred_model': preferred,
        'children_coef': coef,
        'children_pvalue': pval,
        'means_by_children': means_table,
        'interpretation': interpretation
    }

    return results


