from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Helper to find a candidate raw column from several possible names (case-insensitive)
def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair "affairs" dataset into a modeling dataframe.

    Produces these columns (exact names used later in modeling):
      - AffairsCount : numeric version of the original 'affairs' column (coerced)
      - HadAffair : binary outcome (1 = any/high affair, 0 = none/low) - heuristic mapping described below
      - HasChildren : binary indicator of children in the marriage (1=yes, 0=no)
      - Age : numeric derived from 'rating' (kept as-is numeric if possible)
      - IsFemale : best-effort binary female indicator (1=female, 0=male); may be NaN if ambiguous
      - GenderRaw : original gender column preserved (coerced if numeric)
      - YearsMarried, Religiousness, Education, Occupation, MaritalSatisfaction : numeric covariates

    The function attempts to be robust to variations in raw column names by searching
    through plausible alternatives for each conceptual variable.
    """
    df = df.copy()

    # Identify raw columns from plausible alternatives
    col_affairs = _find_column(df, ['affairs', 'Affairs', 'affair', 'numaffairs', 'n_affairs', 'affairs_count'])
    col_children = _find_column(df, ['children', 'child', 'has_children', 'kids', 'children_present'])
    col_rating = _find_column(df, ['rating', 'age', 'Age'])
    col_years = _find_column(df, ['yearsmarried', 'years_married', 'yrs_married', 'yearsmaried'])
    col_relig = _find_column(df, ['religiousness', 'religiosity', 'religious'])
    col_educ = _find_column(df, ['education', 'edu'])
    col_occ = _find_column(df, ['occupation', 'occup', 'job'])
    col_rowname = _find_column(df, ['rownames', 'maritalsatisfaction', 'marital_satisfaction', 'maritalhappiness', 'satisfaction'])
    col_gender = _find_column(df, ['gender', 'sex'])

    # AffairsCount: coerce to numeric from whichever column we found
    if col_affairs is not None:
        df['AffairsCount'] = pd.to_numeric(df[col_affairs], errors='coerce')
    else:
        df['AffairsCount'] = pd.NA

    # Create binary HadAffair using a robust heuristic depending on whether zeros are present
    if 'AffairsCount' in df.columns and df['AffairsCount'].notnull().any():
        try:
            min_val = df['AffairsCount'].min(skipna=True)
            if not pd.isna(min_val) and float(min_val) == 0.0:
                df['HadAffair'] = (df['AffairsCount'] > 0).astype('Int64')
            else:
                med = df['AffairsCount'].median(skipna=True)
                df['HadAffair'] = (df['AffairsCount'] > med).astype('Int64')
        except Exception:
            df['HadAffair'] = pd.NA
    else:
        df['HadAffair'] = pd.NA

    # HasChildren: normalize varieties of yes/no coding from identified raw column
    if col_children is not None:
        raw_children = df[col_children]
    else:
        raw_children = pd.Series([pd.NA] * len(df), index=df.index)

    def map_children(x):
        if pd.isna(x):
            return pd.NA
        # numeric -> treat >0 as having children
        if isinstance(x, (int, float, np.integer, np.floating)):
            try:
                return 1 if x > 0 else 0
            except Exception:
                return pd.NA
        # string -> normalize
        s = str(x).strip().lower()
        if s in ('yes', 'y', 'true', 't', '1', 'yes.'):
            return 1
        if s in ('no', 'n', 'false', 'f', '0', 'no.'):
            return 0
        if 'yes' in s:
            return 1
        if 'no' in s:
            return 0
        # if it looks numeric-like
        try:
            nx = float(s)
            return 1 if nx > 0 else 0
        except Exception:
            return pd.NA

    df['HasChildren'] = raw_children.apply(map_children).astype('Int64')

    # Age: try to use 'rating' or 'age'
    if col_rating is not None:
        df['Age'] = pd.to_numeric(df[col_rating], errors='coerce')
    else:
        df['Age'] = pd.NA

    # Years married
    if col_years is not None:
        df['YearsMarried'] = pd.to_numeric(df[col_years], errors='coerce')
    else:
        df['YearsMarried'] = pd.NA

    # Religiousness
    if col_relig is not None:
        df['Religiousness'] = pd.to_numeric(df[col_relig], errors='coerce')
    else:
        df['Religiousness'] = pd.NA

    # Education and Occupation
    if col_educ is not None:
        df['Education'] = pd.to_numeric(df[col_educ], errors='coerce')
    else:
        df['Education'] = pd.NA

    if col_occ is not None:
        df['Occupation'] = pd.to_numeric(df[col_occ], errors='coerce')
    else:
        df['Occupation'] = pd.NA

    # Marital satisfaction stored in 'rownames' in some schemas
    if col_rowname is not None:
        df['MaritalSatisfaction'] = pd.to_numeric(df[col_rowname], errors='coerce')
    else:
        df['MaritalSatisfaction'] = pd.NA

    # Gender: create GenderRaw and best-effort IsFemale
    if col_gender is not None:
        df['GenderRaw'] = df[col_gender]
    else:
        # preserve the raw column if missing
        df['GenderRaw'] = pd.NA

    # More robust mapping for IsFemale using entire raw gender series
    def _map_is_female_single(val, series):
        if pd.isna(val):
            return pd.NA
        # strings
        if isinstance(val, str):
            s = val.strip().lower()
            # explicit terms
            if s in ('female', 'woman', 'f', 'f.', 'female.', 'female '):
                return 1
            if s in ('male', 'man', 'm', 'm.', 'male.', 'male '):
                return 0
            # contains checks
            if 'female' in s or 'woman' in s:
                return 1
            if 'male' in s or 'man' in s:
                return 0
            # numeric-like strings
            try:
                gn = float(s)
            except Exception:
                return pd.NA
            # fall through to numeric handling below with detection
            val_num = gn
        else:
            # try numeric conversion
            try:
                val_num = float(val)
            except Exception:
                return pd.NA

        # For numeric-like entries, infer mapping from unique numeric codes in the series
        numeric_series = pd.to_numeric(series.dropna(), errors='coerce').dropna()
        unique_vals = set(numeric_series.unique())

        if unique_vals:
            # common codings: 0/1 -> 1=female
            if unique_vals.issubset({0.0, 1.0}):
                return 1 if float(val_num) == 1.0 else 0
            # 1/2 -> 2=female
            if unique_vals.issubset({1.0, 2.0}):
                return 1 if float(val_num) == 2.0 else 0
            # otherwise, if only single unique numeric, try to map that single value to female (conservative)
            if len(unique_vals) == 1:
                single = next(iter(unique_vals))
                # if single == val_num, treat as female, else male (best-effort)
                return 1 if float(val_num) == float(single) else 0

        # Fallback: if val_num is 0 or 1 assume 1 = female
        if float(val_num) in (0.0, 1.0):
            return 1 if float(val_num) == 1.0 else 0

        return pd.NA

    # Apply mapping using the raw gender series to give context for numeric codings
    df['IsFemale'] = df['GenderRaw'].apply(lambda v: _map_is_female_single(v, df['GenderRaw']))

    # At this point we've created the required columns. To reduce downstream dropping of nearly-complete rows,
    # perform mild, conservative imputation for numeric covariates (not for the main IV 'HasChildren' or DV 'HadAffair').
    # Impute numeric control covariates with their medians when available.
    numeric_controls = ['Age', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'MaritalSatisfaction']
    for col in numeric_controls:
        if col in df.columns:
            try:
                median_val = pd.to_numeric(df[col], errors='coerce').median(skipna=True)
                if not pd.isna(median_val):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(median_val)
                else:
                    # leave as-is if median cannot be computed
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.NA

    # For IsFemale, if some mappings exist, fill missing with the modal mapped value; otherwise leave as NA.
    try:
        if df['IsFemale'].notna().any():
            mode_vals = df['IsFemale'].dropna().mode()
            if not mode_vals.empty:
                mode_val = mode_vals.iloc[0]
                df['IsFemale'] = df['IsFemale'].fillna(mode_val)
            # ensure numeric type
            df['IsFemale'] = pd.to_numeric(df['IsFemale'], errors='coerce')
        else:
            # leave as NA (no strong signal)
            df['IsFemale'] = pd.to_numeric(df['IsFemale'], errors='coerce')
    except Exception:
        df['IsFemale'] = pd.to_numeric(df['IsFemale'], errors='coerce')

    # Note: Do not impute HasChildren or HadAffair here to avoid changing primary variables' semantics.
    # But ensure HasChildren column exists and is numeric-compatible
    if 'HasChildren' in df.columns:
        df['HasChildren'] = pd.to_numeric(df['HasChildren'], errors='coerce')
    else:
        df['HasChildren'] = pd.NA

    # Ensure AffairsCount numeric
    df['AffairsCount'] = pd.to_numeric(df['AffairsCount'], errors='coerce')

    # Provide a brief summary column to help debugging downstream (optional)
    try:
        df['_transform_note'] = (
            'AffairsCount_missing=' + df['AffairsCount'].isna().astype(int).astype(str)
            + '; HadAffair_missing=' + df['HadAffair'].isna().astype(int).astype(str)
            + '; HasChildren_missing=' + df['HasChildren'].isna().astype(int).astype(str)
        )
    except Exception:
        df['_transform_note'] = 'transform_complete'

    # Ensure all required final columns exist (even if NA)
    required_final = ['HasChildren', 'HadAffair', 'AffairsCount', 'Age', 'IsFemale', 'GenderRaw',
                      'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'MaritalSatisfaction']
    for rc in required_final:
        if rc not in df.columns:
            df[rc] = pd.NA

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit statistical models to estimate the association between having children and engagement in extramarital affairs.

    Models produced:
      1) Logistic regression (binary outcome HadAffair) using statsmodels.Logit
      2) Poisson regression (count outcome AffairsCount) using statsmodels.GLM with Poisson family

    The function drops rows with missing values in the variables required by each model.
    Returns a dict with fitted result objects for easy further inspection.
    """
    results: Dict[str, Any] = {}

    # Columns required for the models
    base_covs = ['Age', 'IsFemale', 'YearsMarried', 'Religiousness', 'Education', 'Occupation', 'MaritalSatisfaction']

    # Prepare dataset for logistic regression on HadAffair
    model_cols_logit = ['HadAffair', 'HasChildren'] + base_covs
    # Ensure columns exist
    for c in model_cols_logit:
        if c not in df.columns:
            df[c] = pd.NA

    df_logit = df[model_cols_logit].copy()

    # Drop rows with missing values in the key variables: HadAffair and HasChildren.
    # We will impute remaining numeric covariates with their medians if necessary so fitting can proceed.
    df_logit = df_logit.dropna(subset=['HadAffair', 'HasChildren'])

    if df_logit.shape[0] == 0:
        # No usable rows for logistic model; return a note rather than raising an exception.
        results['logit_result'] = None
        results['_logit_note'] = 'Too few complete cases for logistic model after dropping NA: {}'.format(df_logit.shape[0])
    else:
        # Impute remaining numeric covariates (conservative median imputation) to avoid dropping rows unnecessarily.
        for cov in base_covs:
            if cov in df_logit.columns:
                try:
                    if df_logit[cov].isna().any():
                        median_val = pd.to_numeric(df_logit[cov], errors='coerce').median(skipna=True)
                        if not pd.isna(median_val):
                            df_logit[cov] = pd.to_numeric(df_logit[cov], errors='coerce').fillna(median_val)
                        else:
                            df_logit[cov] = pd.to_numeric(df_logit[cov], errors='coerce')
                except Exception:
                    df_logit[cov] = pd.to_numeric(df_logit[cov], errors='coerce')

        # Prepare y and X
        y_logit = pd.to_numeric(df_logit['HadAffair'], errors='coerce')
        X_logit = df_logit[['HasChildren'] + base_covs].astype(float)
        X_logit = sm.add_constant(X_logit, has_constant='add')

        # Drop any rows that still have NA after imputation
        logit_valid_mask = (~y_logit.isna()) & (~X_logit.isna().any(axis=1))
        y_logit = y_logit[logit_valid_mask]
        X_logit = X_logit.loc[logit_valid_mask, :]

        if X_logit.shape[0] == 0:
            results['logit_result'] = None
            results['_logit_note'] = 'No complete cases remain for logistic model after imputation/dropping: 0'
        else:
            # Fit logistic regression (use try/except to catch perfect separation or convergence issues)
            try:
                logit_model = sm.Logit(y_logit, X_logit)
                logit_res = logit_model.fit(disp=False, maxiter=200)
            except Exception:
                # Retry with tiny jitter to break potential perfect separation
                X_logit_jitter = X_logit + np.random.normal(scale=1e-8, size=X_logit.shape)
                logit_model = sm.Logit(y_logit, X_logit_jitter)
                logit_res = logit_model.fit(disp=False, maxiter=200)
            results['logit_result'] = logit_res

    # Prepare dataset for Poisson regression on AffairsCount (counts)
    model_cols_pois = ['AffairsCount', 'HasChildren'] + base_covs
    for c in model_cols_pois:
        if c not in df.columns:
            df[c] = pd.NA

    df_pois = df[model_cols_pois].copy()
    # Drop rows missing the dependent or main IV
    df_pois = df_pois.dropna(subset=['AffairsCount', 'HasChildren'])

    if df_pois.shape[0] == 0:
        results['poisson_result'] = None
        results['_poisson_note'] = 'Too few complete cases for Poisson model after dropping NA: {}'.format(df_pois.shape[0])
    else:
        # Impute numeric covariates with median if needed
        for cov in base_covs:
            if cov in df_pois.columns:
                try:
                    if df_pois[cov].isna().any():
                        median_val = pd.to_numeric(df_pois[cov], errors='coerce').median(skipna=True)
                        if not pd.isna(median_val):
                            df_pois[cov] = pd.to_numeric(df_pois[cov], errors='coerce').fillna(median_val)
                        else:
                            df_pois[cov] = pd.to_numeric(df_pois[cov], errors='coerce')
                except Exception:
                    df_pois[cov] = pd.to_numeric(df_pois[cov], errors='coerce')

        y_pois = pd.to_numeric(df_pois['AffairsCount'], errors='coerce')
        X_pois = df_pois[['HasChildren'] + base_covs].astype(float)
        X_pois = sm.add_constant(X_pois, has_constant='add')

        # Drop any remaining rows with NA
        valid_mask = (~y_pois.isna()) & (~X_pois.isna().any(axis=1))
        y_pois = y_pois[valid_mask]
        X_pois = X_pois.loc[valid_mask, :]

        if X_pois.shape[0] == 0:
            results['poisson_result'] = None
            results['_poisson_note'] = 'No complete cases remain for Poisson model after imputation/dropping: 0'
        else:
            try:
                pois_model = sm.GLM(y_pois, X_pois, family=sm.families.Poisson())
                pois_res = pois_model.fit()
                results['poisson_result'] = pois_res
            except Exception as e:
                results['poisson_result'] = None
                results['_poisson_error'] = str(e)

    # Return fitted model objects and a short summary dictionary with coefficient of HasChildren
    def extract_coef(res, varname='HasChildren'):
        if res is None:
            return None
        try:
            coef = float(res.params[varname])
            pval = float(res.pvalues[varname]) if hasattr(res, 'pvalues') and varname in res.pvalues.index else None
            se = float(res.bse[varname]) if hasattr(res, 'bse') and varname in res.bse.index else None
            return {'coef': coef, 'se': se, 'pvalue': pval}
        except Exception:
            return None

    results['summary'] = {
        'logit_HasChildren': extract_coef(results.get('logit_result')),
        'poisson_HasChildren': extract_coef(results.get('poisson_result'))
    }

    return results