from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/positive_leading_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair Affairs dataset into the analytic dataframe.

    Produces the following columns required by the models:
      - affairs: numeric dependent variable (keeps original coding)
      - children_bin: 1 if 'children' == 'yes', 0 if 'children' == 'no'
      - female: 1 if gender == 'female', 0 if gender == 'male'
      - age, yearsmarried, religiousness, education, occupation, rating: numeric controls

    Drops rows with missing values in any of the variables used in the model.
    """
    df = df.copy()

    # Map children and gender to binary indicators (explicit column names used in models)
    df['children_bin'] = df['children'].map({'yes': 1, 'no': 0})
    df['female'] = df['gender'].map({'female': 1, 'male': 0})

    # Ensure numeric columns are numeric; coerce non-numeric to NaN so we can drop them
    num_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing any of the required variables
    required = ['affairs', 'children_bin', 'female'] + num_cols[1:]
    df = df.dropna(subset=required)

    # Keep only the columns needed for analysis (clean final dataframe)
    final_cols = ['affairs', 'children_bin', 'female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to assess whether having children decreases engagement in extramarital affairs:
      1) OLS regression with robust (HC3) standard errors for a simple, interpretable linear estimate.
      2) Zero-Inflated Negative Binomial (ZINB) count model to account for overdispersion and excess zeros in the 'affairs' outcome.

    Returns a dictionary with model result objects and a short summary (marginal effect for children in the ZINB model).
    """
    results = {}
    df = df.copy()

    # Prepare design matrices
    covariates = ['children_bin', 'female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    X = df[covariates]
    X = sm.add_constant(X)
    y = df['affairs']

    # 1) OLS with robust standard errors
    ols_model = sm.OLS(y, X)
    ols_res = ols_model.fit(cov_type='HC3')
    results['ols_result'] = ols_res

    # 2) Zero-Inflated Negative Binomial (ZINB)
    # Use same regressors for count and inflation parts as a baseline specification.
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    except Exception as e:
        # If import fails, raise with informative message
        raise

    zinb_mod = ZeroInflatedNegativeBinomialP(endog=y, exog=X, exog_infl=X, inflation='logit')

    # Fit the ZINB model. Try Newton first, fall back to BFGS if necessary.
    try:
        zinb_res = zinb_mod.fit(method='newton', maxiter=100, disp=False)
    except Exception:
        zinb_res = zinb_mod.fit(method='bfgs', maxiter=200, disp=False)

    results['zinb_result'] = zinb_res

    # Compute average marginal effect of children (dE[affairs]/d(children_bin)) from ZINB if available
    try:
        # get_margeff sometimes requires specific options; use overall marginal effect and treat children_bin as dummy
        me = zinb_res.get_margeff(at='overall', method='dydx', dummy=True)
        # Extract marginal effect row for children_bin if present
        me_summary = me.summary_frame()
        if 'children_bin' in me_summary.index:
            results['zinb_margeff_children'] = me_summary.loc['children_bin'].to_dict()
        else:
            # If indexing different, attempt to find by variable order
            results['zinb_margeff_children'] = me_summary.iloc[0].to_dict()
    except Exception:
        # If marginal effects computation fails, provide coefficient and basic IRR approximation (exp(coef)) for the count part
        try:
            coef_name = 'children_bin'
            # In ZINB params, the first len(X.columns) are count model params; locate children_bin coefficient
            params = zinb_res.params
            if coef_name in params.index:
                coef = params.loc[coef_name]
                irr = float(np.exp(coef))
                results['zinb_children_coef'] = float(coef)
                results['zinb_children_irr_approx'] = irr
            else:
                results['zinb_children_coef'] = None
        except Exception:
            results['zinb_children_coef'] = None

    # Return both fitted models and computed summaries. Callers can inspect .summary() on the returned result objects.
    return results


