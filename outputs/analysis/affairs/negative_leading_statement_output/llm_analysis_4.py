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
    Transform the raw Fair affairs dataset into an analysis-ready dataframe.

    Steps:
    - Ensure key columns are present and cast to numeric where appropriate.
    - Create binary indicators for children and female.
    - Drop rows with missing values in the dependent variable or any variables used in the model.
    - Return the transformed dataframe containing the exact column names used in modeling.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure dependent variable is numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Map children to binary indicator: 1 = yes, 0 = no
    if 'children' in df.columns:
        df['Children'] = df['children'].map({'yes': 1, 'no': 0})
    else:
        # if column missing, create NA column to fail later
        df['Children'] = np.nan

    # Map gender to Female indicator: 1 = female, 0 = male (handles capitalization)
    if 'gender' in df.columns:
        df['Female'] = df['gender'].astype(str).str.lower().map({'female': 1, 'male': 0})
    else:
        df['Female'] = np.nan

    # Ensure numeric controls are numeric
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # Drop rows with missing values in any columns that will be used in modeling
    required_cols = ['affairs', 'Children', 'Female'] + numeric_cols
    df = df.dropna(subset=required_cols)

    # Optionally cast types
    df['affairs'] = df['affairs'].astype(float)
    df['Children'] = df['Children'].astype(int)
    df['Female'] = df['Female'].astype(int)

    # Return only columns necessary for analysis (keeps original 'affairs')
    return df[['affairs', 'Children', 'Female'] + numeric_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit several models to estimate the association between having children and the count of extramarital affairs.

    Models fitted:
    1) OLS (for comparison; treats affairs as continuous)
    2) Poisson GLM (count model)
    3) Negative Binomial GLM (accounts for overdispersion)
    4) Zero-Inflated Negative Binomial (accounts for excess zeros)

    Returns a dictionary with fitted results and textual summaries for easy inspection.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    results = {}

    # Ensure transform has been applied (columns present)
    required = ['affairs', 'Children', 'Female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula
    formula = 'affairs ~ Children + Female + age + yearsmarried + religiousness + education + occupation + rating'

    # 1) OLS
    try:
        ols_model = smf.ols(formula, data=df).fit()
        results['ols_summary'] = ols_model.summary().as_text()
        results['ols_model'] = ols_model
    except Exception as e:
        results['ols_error'] = str(e)

    # Prepare X and y for GLM and count models
    y = df['affairs']
    X = df[['Children', 'Female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']]
    X = sm.add_constant(X, has_constant='add')

    # 2) Poisson (GLM)
    try:
        poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        results['poisson_summary'] = poisson_model.summary().as_text()
        results['poisson_model'] = poisson_model
    except Exception as e:
        results['poisson_error'] = str(e)

    # 3) Negative Binomial (GLM)
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
        results['nb_summary'] = nb_model.summary().as_text()
        results['nb_model'] = nb_model
    except Exception as e:
        results['nb_error'] = str(e)

    # 4) Zero-Inflated Negative Binomial (more flexible for excess zeros)
    try:
        # For ZeroInflatedNegativeBinomialP we pass exog and exog_infl (use same covariates for inflation by default)
        zinb = ZeroInflatedNegativeBinomialP(endog=y, exog=X, exog_infl=X, inflation='logit')
        zinb_results = zinb.fit(disp=0)
        results['zinb_summary'] = zinb_results.summary().as_text()
        results['zinb_model'] = zinb_results
    except Exception as e:
        # some environments may not have this class or fitting may fail; capture the error
        results['zinb_error'] = str(e)

    # For quick inference about the main question, add a concise summary line if available
    def extract_coef_text(mod, var='Children'):
        try:
            params = mod.params
            pvalues = mod.pvalues
            coef = params.get(var, None)
            pval = pvalues.get(var, None)
            return {'coef': float(coef) if coef is not None else None, 'pvalue': float(pval) if pval is not None else None}
        except Exception:
            return {'coef': None, 'pvalue': None}

    # Collect main coefficient from available fitted models
    summaries = {}
    for k in ['ols_model', 'poisson_model', 'nb_model', 'zinb_model']:
        if k in results and results[k] is not None:
            summaries[k + '_Children'] = extract_coef_text(results[k], var='Children')
    results['Children_coef_summary'] = summaries

    return results


