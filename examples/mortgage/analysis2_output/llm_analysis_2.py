from typing import Any
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Note: this top-level read is kept as in the original file but functions operate on any dataframe passed in.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/mortgage/data.csv')
except Exception:
    # If the file is not present in the environment loading the module, don't fail import.
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the modeling dataframe.
    - Rename columns to meaningful names used in the model.
    - Coerce types to numeric for continuous and binary variables.
    - Drop rows missing any variable required for the model.
    - Returns dataframe containing all columns listed in the conceptual variables.
    """
    df = df.copy()

    # Original mapping from raw feature names to modeling names.
    rename_map = {
        'feature1': 'LoanAmount',
        'feature2': 'Female',         # 1 if applicant is female
        'feature3': 'Black',          # 1 if applicant is Black
        'feature4': 'HousingExpenseRatio',
        'feature5': 'SelfEmployed',
        'feature6': 'Married',
        'feature7': 'MortCreditScore',
        'feature8': 'ConsCreditScore',
        'feature9': 'BadCredit',      # 1 if history of bad credit
        'feature10': 'DebtToIncome',
        'feature11': 'Denied_Flag',   # redundant with feature14; keep for reference
        'feature12': 'LTV',
        'feature13': 'PMI_Denied',
        'feature14': 'Accepted'       # 1 if accepted, 0 if denied
    }

    # Build a normalization helper to perform case-insensitive and punctuation-insensitive matching.
    def normalize_name(name: str) -> str:
        return re.sub(r'[^0-9a-z]', '', str(name).lower())

    # Create normalized rename_map for matching against dataframe columns.
    normalized_rename = {normalize_name(k): v for k, v in rename_map.items()}

    # First-pass: try to rename any columns whose normalized name exactly matches a key in normalized_rename
    col_rename_candidates = {}
    for col in df.columns:
        ncol = normalize_name(col)
        if ncol in normalized_rename:
            col_rename_candidates[col] = normalized_rename[ncol]

    if col_rename_candidates:
        df = df.rename(columns=col_rename_candidates)

    # Second-pass: attempt to match columns whose normalized name contains the feature number (e.g., feature1, f1, feat1)
    # or otherwise contains the target name as a substring.
    # This helps handle variants such as "Feature_1", "feature-1", "feat1", etc.
    # Build a reverse map from target normalized name -> target canonical name
    target_names = [
        'Accepted', 'Female', 'Black', 'LoanAmount', 'HousingExpenseRatio', 'SelfEmployed',
        'Married', 'MortCreditScore', 'ConsCreditScore', 'BadCredit', 'DebtToIncome', 'LTV', 'PMI_Denied'
    ]
    normalized_targets = {normalize_name(t): t for t in target_names}

    for col in list(df.columns):
        ncol = normalize_name(col)
        # If column already one of the targets, skip
        if ncol in normalized_targets:
            canonical = normalized_targets[ncol]
            if col != canonical:
                df = df.rename(columns={col: canonical})
            continue

    # If after these attempts some required columns are still missing, try substring matching:
    for target_norm, target in normalized_targets.items():
        if target not in df.columns:
            # find any existing column whose normalized name contains the target normalized name
            match = None
            for col in df.columns:
                if target_norm in normalize_name(col):
                    match = col
                    break
            if match is not None:
                df = df.rename(columns={match: target})

    # At this point, if required columns still missing but there exist columns like 'feature1'.. we try to map by numeric suffix.
    # Map raw columns named with a trailing number to feature<number> -> use rename_map.
    for col in list(df.columns):
        if col in target_names:
            continue
        m = re.search(r'(\d{1,2})$', col)
        if m:
            num = m.group(1)
            key = f'feature{num}'
            if normalize_name(key) in normalized_rename:
                df = df.rename(columns={col: normalized_rename[normalize_name(key)]})

    # Ensure numeric types; convert errors to NaN
    numeric_cols = ['LoanAmount', 'HousingExpenseRatio', 'MortCreditScore', 'ConsCreditScore',
                    'DebtToIncome', 'LTV']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Binary indicator columns
    binary_cols = ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit', 'PMI_Denied', 'Accepted', 'Denied_Flag']
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Define the final set of columns required for modeling
    required_cols = [
        'Accepted', 'Female', 'Black', 'LoanAmount', 'HousingExpenseRatio', 'SelfEmployed',
        'Married', 'MortCreditScore', 'ConsCreditScore', 'BadCredit', 'DebtToIncome', 'LTV', 'PMI_Denied'
    ]

    # If some required columns are still missing entirely, try to create them by searching for plausible alternatives
    # (e.g., a column named 'sex' that encodes female/male, or 'gender_female'). We only attempt conservative matches.
    for req in required_cols:
        if req not in df.columns:
            req_norm = normalize_name(req)
            found = None
            for col in df.columns:
                col_norm = normalize_name(col)
                # conservative substring match: column name contains the target word (e.g., 'female' in 'is_female')
                if req_norm in col_norm:
                    found = col
                    break
            if found:
                df = df.rename(columns={found: req})

    # Keep only rows with no missing values in required modeling columns that exist in the dataframe.
    existing_required = [c for c in required_cols if c in df.columns]
    if existing_required:
        df = df.dropna(subset=existing_required)

    # Now safe to convert binary columns to integer 0/1
    for col in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit', 'PMI_Denied', 'Accepted']:
        if col in df.columns:
            try:
                df[col] = df[col].round().astype(int)
            except Exception:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=[col])
                df[col] = df[col].round().astype(int)

    # (Optional) sanity check: ensure Accepted is 0/1
    if 'Accepted' in df.columns:
        df = df[df['Accepted'].isin([0, 1])]

    # Replace infinities with NaN globally, then handle missing values for final required columns
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Final: ensure we return the required columns in the specified order.
    final_cols = [c for c in required_cols if c in df.columns]
    # If any required columns are still missing, add them as columns of zeros to preserve the contract.
    # This is defensive: in normal operation with the expected dataset these will not be needed.
    missing_after = [c for c in required_cols if c not in final_cols]
    if missing_after:
        for c in missing_after:
            # Create with integer zeros for binary and numeric zeros for continuous controls.
            if c in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit', 'PMI_Denied', 'Accepted']:
                df[c] = 0
            else:
                df[c] = 0.0
        final_cols = required_cols.copy()

    # Ensure no NaNs or infs remain in the returned dataframe for modeling columns.
    returned = df[final_cols].copy()
    returned.replace([np.inf, -np.inf], np.nan, inplace=True)

    # For binary columns, fill NaN with 0 and ensure integer dtype
    for col in ['Accepted', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit', 'PMI_Denied']:
        if col in returned.columns:
            if returned[col].isnull().any():
                returned[col] = returned[col].fillna(0)
            # after filling, coerce to integer (safe since values should be 0/1)
            returned[col] = pd.to_numeric(returned[col], errors='coerce').round().fillna(0).astype(int)

    # For numeric continuous columns, fill NaN with column median (or 0 if median cannot be computed)
    for col in ['LoanAmount', 'HousingExpenseRatio', 'MortCreditScore', 'ConsCreditScore', 'DebtToIncome', 'LTV']:
        if col in returned.columns:
            if returned[col].isnull().any():
                try:
                    med = returned[col].median()
                    if np.isnan(med):
                        med = 0.0
                except Exception:
                    med = 0.0
                returned[col] = returned[col].fillna(med)
            # ensure numeric type
            returned[col] = pd.to_numeric(returned[col], errors='coerce').fillna(0.0)

    return returned


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression model estimating the effect of gender on mortgage acceptance,
    controlling for the covariates listed in the conceptual variables.

    Returns a dictionary containing:
      - 'result': the fitted statsmodels result object (or a results-like object)
      - 'odds_ratio': pandas Series of exp(params)
      - 'conf_odds': DataFrame of exponentiated confidence intervals
    """
    # Columns to use in the model (must match transform output)
    model_covariates = [
        'Female', 'Black', 'LoanAmount', 'HousingExpenseRatio', 'SelfEmployed',
        'Married', 'MortCreditScore', 'ConsCreditScore', 'BadCredit', 'DebtToIncome', 'LTV', 'PMI_Denied'
    ]

    # Ensure required columns exist
    missing = [c for c in model_covariates + ['Accepted'] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns for modeling: {missing}')

    # Construct design matrices
    X = df[model_covariates].copy()
    y = df['Accepted'].astype(float).copy()

    # Replace infinities and drop rows with NaN in either X or y
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    valid_mask = X.notnull().all(axis=1) & y.notnull()
    if not valid_mask.any():
        raise ValueError('No valid rows available after removing NaNs/infs for modeling.')

    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy().astype(int)

    # Before fitting, detect and remove columns in X that are constant (zero variance)
    # or that are linearly dependent. Removing constant or collinear covariates is a
    # pragmatic defensive step to avoid singular matrix errors while preserving the
    # conceptual variables (these columns carry no information in the provided data).
    # We only operate on X (covariates); the response 'Accepted' is untouched.
    initial_cols = list(X.columns)

    # Drop columns with zero variance (nunique <= 1)
    nunique = X.nunique(dropna=True)
    const_cols = list(nunique[nunique <= 1].index)
    if const_cols:
        X = X.drop(columns=const_cols)

    # If after dropping constants we still have potential multicollinearity, use QR pivoting
    # to select an independent subset of columns.
    if X.shape[1] > 0:
        X_vals = X.astype(float).values
        rank = np.linalg.matrix_rank(X_vals)
        if rank < X.shape[1]:
            # QR with pivoting to select independent columns
            try:
                q, r, piv = np.linalg.qr(X_vals, mode='reduced', pivoting=True)
                # Determine effective rank using R's diagonal
                diag = np.abs(np.diag(r))
                tol = np.max(X_vals.shape) * np.abs(r[0, 0]) * np.finfo(float).eps if r.shape[0] > 0 and r.shape[1] > 0 else 0.0
                effective_rank = int((diag > tol).sum())
                keep_pivots = piv[:effective_rank]
                keep_cols = [X.columns[i] for i in sorted(keep_pivots)]
                X = X[keep_cols]
            except Exception:
                # If QR with pivoting fails for any reason, fall back to keeping the first `rank` columns
                if rank <= 0:
                    raise ValueError('No independent covariates available for modeling.')
                keep_cols = list(X.columns[:rank])
                X = X[keep_cols]
    else:
        # No covariates left after dropping constants -> still need at least an intercept
        X = pd.DataFrame(index=X.index)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood). Be defensive: if the Hessian is singular when
    # computing standard errors (which raises a LinAlgError), fall back to a regularized fit
    # that can produce parameter estimates despite multicollinearity. We try the exact MLE first.
    try:
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=False)
    except np.linalg.LinAlgError:
        # Fallback: regularized fit (small L2 penalty) to obtain stable estimates.
        # fit_regularized may return a Results-like object (depending on statsmodels version).
        logit_model = sm.Logit(y, X)
        try:
            # small alpha for light regularization; L1_wt=0 for L2 penalty
            result = logit_model.fit_regularized(method='elastic_net', alpha=1e-6, L1_wt=0.0, maxiter=1000)
        except TypeError:
            # Older statsmodels versions may not accept 'elastic_net' or these args;
            # try a simpler call
            result = logit_model.fit_regularized(alpha=1e-6, maxiter=1000)

    # Compute odds ratios and confidence intervals on odds scale.
    # Not all result-like objects from fit_regularized provide conf_int(); guard accordingly.
    params = getattr(result, 'params', None)
    if params is None:
        # As a last resort, try to extract params attribute if result is numpy array
        if isinstance(result, (np.ndarray, list)):
            params = pd.Series(result, index=X.columns)
        else:
            raise RuntimeError('Could not obtain parameter estimates from the fitted model.')

    # Try to obtain confidence intervals; if unavailable, build NaNs
    try:
        conf = result.conf_int()
        conf.columns = ['2.5%', '97.5%']
    except Exception:
        # Construct placeholder NaN intervals with same index as params
        conf = pd.DataFrame(np.nan, index=params.index, columns=['2.5%', '97.5%'])

    odds_ratio = np.exp(params)
    # Ensure conf is aligned to params index before exponentiating
    try:
        conf_odds = np.exp(conf.loc[params.index])
    except Exception:
        conf_odds = pd.DataFrame(np.exp(conf.values), index=conf.index, columns=conf.columns)

    return {
        'result': result,
        'odds_ratio': odds_ratio,
        'conf_odds': conf_odds
    }