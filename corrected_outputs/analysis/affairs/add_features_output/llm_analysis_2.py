from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure key columns exist
    required = ['affairs', 'children']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    # Drop rows with missing DV or children indicator
    df = df.dropna(subset=['affairs', 'children'])

    # Create binary children indicator (1 = yes, 0 = no). Handle capitalization/whitespace.
    df['children_binary'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    # If mapping produced NaN (unexpected values), mark those rows for removal
    df = df.dropna(subset=['children_binary'])
    df['children_binary'] = df['children_binary'].astype(int)

    # Create gender binary (female = 1, male = 0). If gender missing/unexpected, will be coerced to NaN and dropped later.
    if 'gender' in df.columns:
        df['gender_female'] = df['gender'].astype(str).str.strip().str.lower().map({'female': 1, 'male': 0})
    else:
        # If gender not present, create NA column (will be dropped if used in model)
        df['gender_female'] = np.nan

    # Coerce numeric control columns to numeric, preserving NaN for invalid parsing
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'affairs']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing any variable that will be used in the model (DV, IV, and controls)
    model_cols = ['affairs', 'children_binary', 'gender_female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    # Keep only columns that actually exist in df to avoid KeyError; but we require at least the ones specified in cvars
    existing_model_cols = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=existing_model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits two specifications to assess the relationship between having children and number of affairs:
    1) OLS as a simple baseline
    2) A zero-inflated negative binomial (preferred for overdispersed counts with excess zeros).

    Returns a dict with keys 'ols' and 'zinb' containing fitted model results objects.
    If ZINB fails to converge or is unavailable, falls back to Zero-Inflated Poisson.
    """
    import statsmodels.api as sm
    from statsmodels.tools import add_constant
    # Import count models; these are available in statsmodels >= 0.10
    try:
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson
    except Exception:
        # If import fails, raise a clear error
        raise ImportError("Required Zero-Inflated count models not available in this statsmodels installation.")

    # Define regressors (must match transformed dataframe columns specified above)
    X_cols = ['children_binary', 'gender_female', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    # Keep only columns that exist in df (transform should have produced them); this protects against unexpected schemas
    X_cols = [c for c in X_cols if c in df.columns]
    if 'affairs' not in df.columns:
        raise ValueError("Dependent variable 'affairs' not found in dataframe")

    X = df[X_cols]
    X = add_constant(X, has_constant='add')
    y = df['affairs']

    # 1) OLS baseline
    ols_mod = sm.OLS(y, X)
    ols_res = ols_mod.fit()

    # 2) Zero-inflated Negative Binomial (or ZIP fallback)
    zinb_res = None
    try:
        zinb_mod = ZeroInflatedNegativeBinomialP(endog=y, exog=X, exog_infl=X, inflation='logit')
        zinb_res = zinb_mod.fit(disp=False)
    except Exception as e:
        # Try ZIP fallback
        try:
            zip_mod = ZeroInflatedPoisson(endog=y, exog=X, exog_infl=X, inflation='logit')
            zinb_res = zip_mod.fit(disp=False)
        except Exception as e2:
            # If both fail, raise the original error to aid debugging
            raise RuntimeError(f"ZINB and ZIP model fitting failed. ZINB error: {e}; ZIP error: {e2}")

    # Return the fitted model result objects. The caller can inspect summary() on each.
    return {
        'ols': ols_res,
        'zinb_or_zip': zinb_res,
        'model_spec': {
            'exog_variables': X_cols,
            'dv': 'affairs'
        }
    }


