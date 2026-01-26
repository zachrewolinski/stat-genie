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
    Transform the raw Fair (1978) dataset into an analysis-ready dataframe.

    Produces the following columns used in modeling:
      - affairs: numeric count of extramarital affairs (keeps original numeric values)
      - has_children: binary indicator (1 if 'children' == 'yes', 0 if 'no')
      - female: binary indicator for gender (1 if female, 0 if male)
      - age, yearsmarried, religiousness, education, occupation, rating: numeric controls

    Drops rows with missing values in any of the above columns.
    """
    df = df.copy()

    # Normalize string fields and create binary indicators
    if 'children' in df.columns:
        df['children'] = df['children'].astype(str).str.strip().str.lower()
        df['has_children'] = df['children'].map({'yes': 1, 'no': 0})
    else:
        df['has_children'] = np.nan

    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.strip().str.lower()
        df['female'] = df['gender'].map({'female': 1, 'male': 0})
    else:
        df['female'] = np.nan

    # Ensure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Required columns for modeling
    required = ['affairs', 'has_children', 'female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    # Drop rows with missing values in any required columns
    df = df.dropna(subset=required).reset_index(drop=True)

    # Keep only the columns needed for modeling (cleaner output)
    final_cols = ['affairs', 'has_children', 'female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df[final_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit several models to estimate the association between having children and reported extramarital affairs.

    Models fit:
      1) Negative binomial regression (accounts for overdispersion in count data) using GLM negative binomial family
      2) Zero-inflated negative binomial (accounts for excess zeros) if available
      3) OLS with robust (HC3) standard errors for a simple comparison

    Also computes group means and a Welch t-test between parents and non-parents as a descriptive check.

    Returns a dict with fitted model result objects and descriptive statistics.
    """
    from scipy import stats

    # Try to import ZeroInflatedNegativeBinomialP if available; if not, we'll record an error for that model.
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
        _has_zinb = True
    except Exception:
        ZeroInflatedNegativeBinomialP = None  # type: ignore
        _has_zinb = False

    # Columns used in modeling
    model_cols = ['affairs', 'has_children', 'female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df_model = df[model_cols].dropna().copy()

    y = df_model['affairs']
    X = df_model[['has_children', 'female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']]
    X = sm.add_constant(X, has_constant='add')

    results = {}

    # 1) Negative Binomial via GLM (uses the NegativeBinomial family)
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
        results['negative_binomial'] = nb_res
    except Exception as e:
        results['negative_binomial_error'] = str(e)

    # 2) Zero-inflated Negative Binomial (logit inflation); use a small inflation model (intercept + has_children + female)
    try:
        if _has_zinb and ZeroInflatedNegativeBinomialP is not None:
            exog_infl = sm.add_constant(df_model[['has_children', 'female']], has_constant='add')
            zinb_model = ZeroInflatedNegativeBinomialP(endog=y, exog=X, exog_infl=exog_infl, inflation='logit')
            zinb_res = zinb_model.fit(disp=False, maxiter=200)
            results['zero_inflated_negative_binomial'] = zinb_res
        else:
            results['zero_inflated_negative_binomial_error'] = 'ZeroInflatedNegativeBinomialP not available in this statsmodels installation.'
    except Exception as e:
        results['zero_inflated_negative_binomial_error'] = str(e)

    # 3) OLS with robust standard errors (HC3) as a simple baseline
    try:
        ols_model = sm.OLS(y, X)
        ols_res_raw = ols_model.fit()
        # Attach HC3 robust covariance results
        ols_res = ols_res_raw.get_robustcov_results(cov_type='HC3')
        results['ols_robust'] = ols_res
    except Exception as e:
        results['ols_robust_error'] = str(e)

    # Descriptive: group means and Welch t-test
    try:
        group_stats = df_model.groupby('has_children')['affairs'].agg(['mean', 'std', 'count']).to_dict()
        g1 = df_model[df_model['has_children'] == 1]['affairs']
        g0 = df_model[df_model['has_children'] == 0]['affairs']
        ttest = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
        results['group_stats'] = group_stats
        results['welch_ttest'] = {
            'statistic': float(ttest.statistic) if ttest.statistic is not None else None,
            'pvalue': float(ttest.pvalue) if ttest.pvalue is not None else None
        }
    except Exception as e:
        results['group_stats_error'] = str(e)

    # Return fitted results and descriptive checks
    return results