from typing import Any, Dict, List
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError


# Helper: candidate names for possible raw columns that map to required final columns
_CANDIDATES: Dict[str, List[str]] = {
    'FocalID': ['feature1', 'focalid', 'focal_id', 'focal', 'group1', 'group_a'],
    'OtherID': ['feature2', 'otherid', 'other_id', 'other', 'group2', 'group_b'],
    'DyadID': ['feature3', 'dyadid', 'dyad_id', 'pair_id', 'pair'],
    'FocalWon': ['feature4', 'focalwon', 'focal_won', 'winner', 'outcome', 'won', 'focal_winner'],
    'FocalDist': ['feature5', 'focaldist', 'focal_dist', 'dist_focal', 'distance_focal'],
    'OtherDist': ['feature6', 'otherdist', 'other_dist', 'dist_other', 'distance_other'],
    'FocalSize': ['feature7', 'focalsize', 'focal_size', 'size_focal', 'n_focal'],
    'OtherSize': ['feature8', 'othersize', 'other_size', 'size_other', 'n_other'],
    'FocalMales': ['feature9', 'focalmales', 'focal_males', 'males_focal', 'n_males_focal'],
    'OtherMales': ['feature10', 'othermales', 'other_males', 'males_other', 'n_males_other'],
    'FocalFemales': ['feature11', 'focalfemales', 'focal_females', 'females_focal', 'n_females_focal'],
    'OtherFemales': ['feature12', 'otherfemales', 'other_females', 'females_other', 'n_females_other'],
}


def _find_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the dataframe has columns with the exact required final names.
    For each required final column, if it's not already present, look for plausible
    candidate source columns (case-insensitive). If found, rename the first match.
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    for required, candidates in _CANDIDATES.items():
        if required in df.columns:
            continue
        found = None
        for cand in candidates:
            # check exact match first
            if cand in df.columns:
                found = cand
                break
            # case-insensitive search
            low = cand.lower()
            if low in cols_lower:
                found = cols_lower[low]
                break
        if found:
            df = df.rename(columns={found: required})
            # update lowercase mapping for subsequent iterations
            cols_lower = {c.lower(): c for c in df.columns}
    return df


def _infer_focal_won(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to infer/create the FocalWon column when it is not present.
    Strategies (in order):
      1. Find a numeric 0/1 column and use it.
      2. If a column contains per-row identifiers matching either FocalID or OtherID,
         interpret matches to FocalID as focal wins.
      3. If a column contains strings like 'focal'/'other' or 'f'/'o'/'won'/'lost',
         map focal-like strings to 1, others to 0.
    Returns a copy of df (possibly modified). Does not raise; callers should check
    whether FocalWon was created.
    """
    df = df.copy()

    if 'FocalWon' in df.columns:
        return df

    # Helper to test for numeric 0/1 column
    for col in df.columns:
        if col == 'FocalWon':
            continue
        # try numeric conversion
        s_num = pd.to_numeric(df[col], errors='coerce')
        non_na = s_num.dropna()
        if len(non_na) == 0:
            continue
        unique_vals = set(non_na.unique().tolist())
        if unique_vals.issubset({0.0, 1.0}):
            # Use this column as FocalWon
            df['FocalWon'] = s_num.fillna(0).astype(int)
            return df

    # If FocalID and OtherID present, look for a column containing per-row identifiers
    if ('FocalID' in df.columns) and ('OtherID' in df.columns):
        f_str = df['FocalID'].astype(str)
        o_str = df['OtherID'].astype(str)
        for col in df.columns:
            if col in {'FocalID', 'OtherID', 'DyadID'}:
                continue
            s = df[col].astype(str)
            # Check if every non-missing entry equals either focal or other for that row
            mask_valid = (~df[col].isna()) & ((s == f_str) | (s == o_str))
            # If all non-missing entries match either focal or other, accept
            if mask_valid.sum() == (~df[col].isna()).sum() and mask_valid.sum() > 0:
                df['FocalWon'] = (s == f_str).astype(int)
                return df

    # String label mapping: map focal-like values to 1, others to 0
    focal_like = {'focal', 'f', 'winner', 'won', 'yes', 'y', 'true', 't', '1'}
    # Remove ambiguous single-letter token 'f' from other_like to avoid conflict
    other_like = {'other', 'o', 'loser', 'lost', 'no', 'n', 'false', '0'}
    for col in df.columns:
        if col in {'FocalID', 'OtherID', 'DyadID'}:
            continue
        non_null = df[col].dropna().astype(str).str.lower()
        if len(non_null) == 0:
            continue
        # Collect unique tokens (trim whitespace)
        tokens = set(non_null.str.strip().unique().tolist())
        # If tokens are a subset of known focal/other-like tokens, map accordingly
        allowed_tokens = focal_like.union(other_like)
        if tokens.issubset(allowed_tokens):
            df['FocalWon'] = df[col].astype(str).str.lower().str.strip().isin(focal_like).astype(int)
            return df

    # If nothing inferred, return unchanged
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.
    Produces the following columns (used by the model):
      - FocalID, OtherID, DyadID, FocalWon,
      - FocalDist, OtherDist, FocalSize, OtherSize,
      - FocalMales, OtherMales, FocalFemales, OtherFemales,
      - SizeRatio, LogSizeRatio, SizeDiff,
      - LocationAdvantage (categorical), LocationAdv_binary (0/1),
      - MalesDiff, FemalesDiff, TotalSize

    The function will try to detect common alternative raw column names and
    rename them to the required canonical names. If critical columns cannot
    be found, a KeyError is raised.
    """
    df = df.copy()

    # Attempt to map alternative column names to the required canonical names
    df = _find_and_rename(df)

    # If FocalWon still missing, try to infer it from other available columns
    if 'FocalWon' not in df.columns:
        df = _infer_focal_won(df)

    # Required raw inputs for creating the final variables
    required_raw = ['FocalWon', 'FocalSize', 'OtherSize', 'FocalDist', 'OtherDist']

    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required input columns for transform: {missing}")

    # Ensure numeric types where expected (apply only to columns that exist)
    numeric_cols = [
        'FocalWon', 'FocalDist', 'OtherDist', 'FocalSize', 'OtherSize',
        'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # If DyadID missing, attempt to build it from FocalID and OtherID
    if 'DyadID' not in df.columns:
        if ('FocalID' in df.columns) and ('OtherID' in df.columns):
            df['DyadID'] = df['FocalID'].astype(str) + '_' + df['OtherID'].astype(str)
        else:
            # If we cannot construct DyadID, create a unique observation-level DyadID
            # This preserves the required column name but means there are no repeated dyads.
            df['DyadID'] = pd.RangeIndex(start=0, stop=len(df), step=1).astype(str)

    # Drop rows with missing critical information that we cannot compute without
    df = df.dropna(subset=['FocalWon', 'FocalSize', 'OtherSize', 'FocalDist', 'OtherDist'])

    # Derived group size measures
    eps = 1e-9
    df['SizeRatio'] = df['FocalSize'] / (df['OtherSize'] + eps)
    df['LogSizeRatio'] = np.log(df['SizeRatio'] + eps)
    df['SizeDiff'] = df['FocalSize'] - df['OtherSize']

    # Location advantage: binary indicator and categorical label
    df['LocationAdvantage'] = np.where(
        df['FocalDist'] < df['OtherDist'],
        'FocalHome',
        np.where(df['FocalDist'] > df['OtherDist'], 'OtherHome', 'Neutral')
    )
    df['LocationAdv_binary'] = (df['FocalDist'] < df['OtherDist']).astype(int)

    # Sex-composition and size controls
    # If male/female count columns are missing, fill with zeros (conservative)
    if 'FocalMales' not in df.columns:
        df['FocalMales'] = 0
    if 'OtherMales' not in df.columns:
        df['OtherMales'] = 0
    if 'FocalFemales' not in df.columns:
        df['FocalFemales'] = 0
    if 'OtherFemales' not in df.columns:
        df['OtherFemales'] = 0

    df['MalesDiff'] = df['FocalMales'] - df['OtherMales']
    df['FemalesDiff'] = df['FocalFemales'] - df['OtherFemales']
    df['TotalSize'] = df['FocalSize'] + df['OtherSize']

    # Cast DyadID to categorical for later inclusion as fixed effects
    df['DyadID'] = df['DyadID'].astype('category')

    # Ensure binary dependent variable is integer 0/1
    # If values are not 0/1, attempt to coerce: treat nonzero as 1
    df['FocalWon'] = pd.to_numeric(df['FocalWon'], errors='coerce')
    df = df.dropna(subset=['FocalWon'])
    df['FocalWon'] = (df['FocalWon'] != 0).astype(int)

    # Final minimal NA drop for newly created columns required by the model
    final_required = ['LogSizeRatio', 'LocationAdv_binary', 'MalesDiff', 'FemalesDiff', 'TotalSize']
    df = df.dropna(subset=final_required)

    # Keep only columns that are relevant for downstream analysis plus identifiers
    keep_cols = [
        'FocalID', 'OtherID', 'DyadID', 'FocalWon',
        'FocalDist', 'OtherDist', 'FocalSize', 'OtherSize',
        'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales',
        'SizeRatio', 'LogSizeRatio', 'SizeDiff',
        'LocationAdvantage', 'LocationAdv_binary',
        'MalesDiff', 'FemalesDiff', 'TotalSize'
    ]
    # Only keep those that exist to avoid KeyError
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression model predicting the probability that the focal group won.

    Model specification (adaptive):
      Base predictors: FocalWon ~ LogSizeRatio * LocationAdv_binary + MalesDiff + FemalesDiff + TotalSize
      If DyadID exhibits repeated observations (i.e., fewer unique DyadID values than rows),
      include C(DyadID) as categorical fixed effects: + C(DyadID)
      If every DyadID is unique (no repeats), omit C(DyadID) to avoid perfect multicollinearity.

    Returns the fitted statsmodels result object (LogitResults) when possible.
    Falls back to a sklearn-based wrapper object if statsmodels estimation repeatedly fails.
    """
    # Work on a copy
    df = df.copy()

    # Validate that required final columns exist
    required_model_cols = ['FocalWon', 'LogSizeRatio', 'LocationAdv_binary', 'MalesDiff', 'FemalesDiff', 'TotalSize', 'DyadID']
    missing = [c for c in required_model_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Ensure types are appropriate
    df['LocationAdv_binary'] = pd.to_numeric(df['LocationAdv_binary'], errors='coerce').fillna(0).astype(int)
    df['MalesDiff'] = pd.to_numeric(df['MalesDiff'], errors='coerce').fillna(0)
    df['FemalesDiff'] = pd.to_numeric(df['FemalesDiff'], errors='coerce').fillna(0)
    df['TotalSize'] = pd.to_numeric(df['TotalSize'], errors='coerce').fillna(0)
    df['LogSizeRatio'] = pd.to_numeric(df['LogSizeRatio'], errors='coerce').fillna(0)
    df['FocalWon'] = pd.to_numeric(df['FocalWon'], errors='coerce').fillna(0).astype(int)

    # Build base formula
    base_terms = 'LogSizeRatio * LocationAdv_binary + MalesDiff + FemalesDiff + TotalSize'

    # Include dyad fixed effects only if there are repeated observations for some dyads.
    include_dyad = df['DyadID'].nunique() < len(df)
    if include_dyad:
        formula_full = f'FocalWon ~ {base_terms} + C(DyadID)'
    else:
        formula_full = f'FocalWon ~ {base_terms}'

    # Try to fit using statsmodels with a sequence of fallbacks to avoid singular matrix errors.
    # Preferred: standard MLE via different optimization methods. Final fallback: sklearn logistic regression.
    solver_candidates = [None, 'newton', 'bfgs', 'lbfgs', 'cg']
    tried_exceptions = []
    # If include_dyad, first try formula_full (with C(DyadID)); otherwise formula_full already reduced.
    for solver in solver_candidates:
        try:
            if solver is None:
                model_result = smf.logit(formula=formula_full, data=df).fit(disp=False)
            else:
                model_result = smf.logit(formula=formula_full, data=df).fit(disp=False, method=solver, maxiter=200)
            return model_result
        except Exception as e:
            tried_exceptions.append(e)
            # If this is a clear perfect separation / singularity, try reduced model (no dyad FE) next.
            msg = str(e).lower()
            if include_dyad and (isinstance(e, (np.linalg.LinAlgError, PerfectSeparationError)) or 'singular' in msg or 'perfect' in msg):
                # Try reduced model without Dyad fixed effects
                reduced_formula = f'FocalWon ~ {base_terms}'
                for solver2 in solver_candidates:
                    try:
                        if solver2 is None:
                            model_result = smf.logit(formula=reduced_formula, data=df).fit(disp=False)
                        else:
                            model_result = smf.logit(formula=reduced_formula, data=df).fit(disp=False, method=solver2, maxiter=200)
                        return model_result
                    except Exception as e2:
                        tried_exceptions.append(e2)
                # If reduced model attempts also fail, break to sklearn fallback
                break
            # Otherwise continue trying other solvers for the same formula
            continue

    # At this point, statsmodels fits have failed repeatedly. Fall back to sklearn logistic regression.
    # Build design matrices via patsy (used by statsmodels) to ensure consistent encoding.
    try:
        from patsy import dmatrices
        y, X = dmatrices(f'FocalWon ~ {base_terms}', data=df, return_type='dataframe')
        # Flatten y to 1d array
        y = np.asarray(y).ravel()
    except Exception:
        # As a very conservative fallback, construct X manually using required columns.
        # This will include intercept implicitly.
        X = pd.DataFrame({
            'Intercept': 1.0,
            'LogSizeRatio': df['LogSizeRatio'],
            'LocationAdv_binary': df['LocationAdv_binary'],
            'LogSizeRatio:LocationAdv_binary': df['LogSizeRatio'] * df['LocationAdv_binary'],
            'MalesDiff': df['MalesDiff'],
            'FemalesDiff': df['FemalesDiff'],
            'TotalSize': df['TotalSize']
        })
        y = df['FocalWon'].values

    # Now fit sklearn logistic regression as a robust fallback
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
        clf.fit(X, y)
        # Build a params Series aligned with X columns
        coef = clf.coef_.ravel()
        intercept = clf.intercept_.ravel()
        params_values = []
        params_index = []
        if 'Intercept' in X.columns:
            params_index.append('Intercept')
            params_values.append(intercept[0])
            feature_cols = [c for c in X.columns if c != 'Intercept']
        else:
            # No explicit intercept column; sklearn's intercept applies separately
            feature_cols = list(X.columns)
            params_index.append('Intercept')
            params_values.append(intercept[0])

        params_index.extend(feature_cols)
        params_values.extend(coef.tolist())
        params_series = pd.Series(data=np.array(params_values), index=params_index)

        # Construct a lightweight result object that mimics some common statsmodels attributes/methods.
        from types import SimpleNamespace

        def predict_proba_func(exog: pd.DataFrame = None):
            if exog is None:
                mat = X
            else:
                # If exog has the same columns as X, use them; otherwise try to select matching columns.
                if set(X.columns).issubset(set(exog.columns)):
                    mat = exog[X.columns]
                else:
                    # attempt to build design matrix via patsy using the same formula
                    try:
                        from patsy import dmatrix
                        mat = dmatrix(f'{base_terms} - 1', exog, return_type='dataframe')
                        # patsy without intercept (-1); add intercept if X had one
                        if 'Intercept' in X.columns:
                            mat = mat.copy()
                            mat.insert(0, 'Intercept', 1.0)
                    except Exception:
                        # last-resort: try to align common columns
                        common = [c for c in X.columns if c in exog.columns]
                        mat = exog[common].reindex(columns=X.columns, fill_value=0.0)
            probs = clf.predict_proba(mat)[:, 1]
            return probs

        def predict_func(exog: pd.DataFrame = None):
            probs = predict_proba_func(exog=exog)
            return (probs >= 0.5).astype(int)

        result_obj = SimpleNamespace(
            params=params_series,
            predict_proba=predict_proba_func,
            predict=predict_func,
            sklearn_model=clf,
            design_info_columns=list(X.columns)
        )
        return result_obj
    except Exception as e_final:
        # If even the sklearn fallback fails, raise the last statsmodels exception to surface original issue.
        # Prefer to raise a clear error rather than returning None.
        raise RuntimeError("All model fitting attempts failed. Last exception: " + repr(e_final)) from e_final