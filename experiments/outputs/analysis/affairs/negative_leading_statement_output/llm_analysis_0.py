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
    Clean and prepare the Fair (1978) affairs dataset for modeling.

    Steps:
    - Standardize and encode 'children' into a binary indicator children_dummy (1 if yes, 0 if no).
    - Encode gender into gender_male (1 if male, 0 if female).
    - Ensure numeric columns are numeric and drop rows missing any variables required for the model.
    - Return a dataframe containing only the columns needed for the statistical models.
    """

    df = df.copy()

    # Standardize column names if necessary (assume given schema already uses these names)
    required_original_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    # If any required columns are missing, raise an informative error
    missing_cols = [c for c in required_original_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataframe is missing required columns: {missing_cols}")

    # Encode children: handle capitalization/whitespace; map 'yes'->1, 'no'->0
    df['children_dummy'] = (
        df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    )

    # Encode gender into a male indicator (1 = male, 0 = female). If other values appear, map to NaN.
    df['gender_male'] = (
        df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    )

    # Ensure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = ['affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in any of the modeling columns
    model_cols = ['affairs', 'children_dummy', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    df = df.dropna(subset=model_cols).copy()

    # Ensure affairs is integer (counts) since the original coding uses numeric codes that are counts/frequencies
    # but keep as numeric (int) for modeling
    df['affairs'] = df['affairs'].astype(int)

    # Reset index for a clean returned dataframe
    df = df.reset_index(drop=True)

    # Return only the columns required for modeling (leave original columns intact in df but return full df with these columns)
    # The model function will select from these columns by name.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two models to estimate association between having children and extramarital affairs:
      1) OLS (linear regression) as a simple baseline.
      2) Zero-inflated negative binomial (ZINB) for count data with excess zeros and overdispersion.

    Returns a dictionary with fitted model results and textual summaries.
    """

    import statsmodels.api as sm
    from statsmodels.tools import add_constant
    # Import zero-inflated count models; fallback to ZeroInflatedPoisson if ZINB is unavailable
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP as ZINB
    except Exception:
        from statsmodels.discrete.count_model import ZeroInflatedPoisson as ZINB  # fallback (name differs but used similarly)

    results = {}

    # Define covariates (controls) and independent variable
    covariate_names = ['children_dummy', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    # Ensure all required columns exist in the dataframe
    missing = [c for c in covariate_names + ['affairs'] if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # Design matrices
    X = df[covariate_names]
    X_const = add_constant(X, has_constant='add')
    y = df['affairs']

    # 1) OLS baseline
    ols_model = sm.OLS(y, X_const)
    ols_res = ols_model.fit()
    results['ols_result'] = ols_res
    # Save readable summary text
    results['ols_summary'] = ols_res.summary().as_text()

    # 2) Zero-inflated negative binomial model
    # Use the same covariates for both the count and the inflation (logit) part for transparency.
    try:
        zinb_model = ZINB(endog=y, exog=X_const, exog_infl=X_const, inflation='logit')
        # fit quietly; allow up to 200 maxiter to improve convergence chances
        zinb_res = zinb_model.fit(disp=False, maxiter=200)
        results['zinb_result'] = zinb_res
        results['zinb_summary'] = zinb_res.summary().as_text()
    except Exception as e:
        # If ZINB fitting fails, attempt Zero-Inflated Poisson as fallback and record the error
        results['zinb_fit_error'] = str(e)
        try:
            from statsmodels.discrete.count_model import ZeroInflatedPoisson
            zip_model = ZeroInflatedPoisson(endog=y, exog=X_const, exog_infl=X_const, inflation='logit')
            zip_res = zip_model.fit(disp=False, maxiter=200)
            results['zip_result'] = zip_res
            results['zip_summary'] = zip_res.summary().as_text()
        except Exception as e2:
            results['zip_fit_error'] = str(e2)

    # Compute and store an interpretable effect estimate for the key variable 'children_dummy'
    # For OLS: direct coefficient and p-value
    if 'ols_result' in results:
        coef = results['ols_result'].params.get('children_dummy', np.nan)
        pval = results['ols_result'].pvalues.get('children_dummy', np.nan)
        results['ols_children_effect'] = {'coef': float(coef), 'pvalue': float(pval)}

    # For count models: extract coefficient in the count model portion (if available).
    # The parameter name for the count model will normally be 'children_dummy' when exog is used for the count part.
    if 'zinb_result' in results:
        try:
            zinb_coef = results['zinb_result'].params.get('children_dummy', np.nan)
            zinb_pval = results['zinb_result'].pvalues.get('children_dummy', np.nan)
            results['zinb_children_effect'] = {'coef': float(zinb_coef), 'pvalue': float(zinb_pval)}
        except Exception:
            results['zinb_children_effect'] = None
    elif 'zip_result' in results:
        try:
            zip_coef = results['zip_result'].params.get('children_dummy', np.nan)
            zip_pval = results['zip_result'].pvalues.get('children_dummy', np.nan)
            results['zip_children_effect'] = {'coef': float(zip_coef), 'pvalue': float(zip_pval)}
        except Exception:
            results['zip_children_effect'] = None

    # Return the results dict. The calling code can inspect summaries and coefficient dictionaries.
    return results


