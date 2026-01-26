from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (1978) affairs dataset into a modeling-ready dataframe.

    Produces the following new columns used in the model:
    - affair_count: integer count of affairs (from 'affairs')
    - children_binary: 1 if children present in marriage, 0 otherwise
    - gender_male: 1 if male, 0 if female
    - age_c, yearsmarried_c, religiousness_c, education_c, occupation_c, rating_c: centered numeric controls

    Drops rows with missing values in the outcome, IV, or key controls.
    """
    df = df.copy()

    # Keep only rows with non-missing outcome and children indicator
    df = df.dropna(subset=['affairs', 'children'])

    # Create a clean integer affair count column
    # The original coding uses numeric values (0,1,2,3,7,12) to reflect categories/frequencies.
    df['affair_count'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Map children to binary -- handle lowercase/uppercase variants
    def _map_children(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ['yes', 'y', '1', 'true']:
            return 1
        if s in ['no', 'n', '0', 'false']:
            return 0
        return np.nan

    df['children_binary'] = df['children'].apply(_map_children)

    # Map gender to binary male/female
    def _map_gender(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s.startswith('m'):
            return 1
        if s.startswith('f'):
            return 0
        return np.nan

    df['gender_male'] = df['gender'].apply(_map_gender)

    # Controls we will require
    required_controls = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    # Convert required controls to numeric where possible
    for col in required_controls:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in the core modeling variables
    core_required = ['affair_count', 'children_binary'] + required_controls
    df = df.dropna(subset=core_required)

    # Ensure affair_count integer and non-negative
    df['affair_count'] = df['affair_count'].astype(int)
    df = df[df['affair_count'] >= 0]

    # Create centered versions of numeric controls for interpretability
    for col in required_controls:
        df[f'{col}_c'] = df[col] - df[col].mean()

    # Cast binary indicators to int
    df['children_binary'] = df['children_binary'].astype(int)
    df['gender_male'] = df['gender_male'].astype(int)

    # Final columns required by the statistical model are kept; others remain but model will only reference the listed ones
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit count regression models to estimate the association between having children and extramarital affairs.

    Primary specification: Negative Binomial regression (accounts for overdispersion relative to Poisson).
    Secondary specification: Zero-Inflated Negative Binomial (attempted) to allow for excess zeros.

    Returns a dictionary with model results objects. The keys are 'nb_model' and (if available) 'zinb_model'.
    """
    import statsmodels.api as sm
    from statsmodels.tools import add_constant

    # Prepare design matrix and outcome variable using the exact transformed column names
    exog_vars = [
        'children_binary',
        'gender_male',
        'age_c',
        'yearsmarried_c',
        'religiousness_c',
        'education_c',
        'occupation_c',
        'rating_c'
    ]

    # Ensure columns present
    for v in exog_vars + ['affair_count']:
        if v not in df.columns:
            raise ValueError(f"Required column '{v}' not found in dataframe")

    X = df[exog_vars]
    X = add_constant(X, has_constant='add')
    y = df['affair_count']

    results = {}

    # 1) Negative binomial via GLM (uses NB likelihood in GLM framework)
    nb_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    nb_res = nb_glm.fit()
    results['nb_model'] = nb_res

    # 2) Try a more flexible discrete-count model: Zero-Inflated Negative Binomial (if available)
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
        # Use the same exog for the count and the inflation model for simplicity
        zinb = ZeroInflatedNegativeBinomialP(endog=y, exog=X, exog_infl=X, inflation='logit')
        zinb_res = zinb.fit(disp=0)
        results['zinb_model'] = zinb_res
    except Exception as e:
        # If ZeroInflatedNegativeBinomialP is unavailable or fails to converge, return the error message
        results['zinb_error'] = str(e)

    # Optionally report a simple summary dictionary of coefficients for quick programmatic access
    coef_summary = nb_res.summary2().tables[1][['Coef.', 'Std.Err.', 'P>|z|']]
    results['nb_coef_table'] = coef_summary

    return results


