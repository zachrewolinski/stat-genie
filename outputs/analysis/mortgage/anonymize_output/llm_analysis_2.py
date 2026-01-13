from typing import Any
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Boston mortgage dataset into a dataframe suitable for modeling.

    Steps performed:
    - Copy dataframe to avoid modifying input in-place.
    - Coerce original feature columns to numeric where present.
    - Derive the outcome Approved using feature14 when available or 1 - feature11 otherwise.
    - Derive binary control variables for demographics and indicators.
    - Keep continuous controls and standardize (z-score) them to aid model convergence and interpretation.
    - Drop rows with missing values in required outcome (Approved).
      For binary controls including Female, any remaining missing values are imputed with 0 before modeling.
    - Impute missing standardized continuous variables with 0 (the standardized mean) to preserve observations for modeling.
    - Return dataframe containing only the final columns (both derived binary and standardized continuous variables).
    """
    df = df.copy()

    # Helper to robustly parse binary-like series with token sets
    def parse_binary(col_series: pd.Series, positive_tokens: set, negative_tokens: set = None) -> pd.Series:
        """
        Attempt to parse a pandas Series into numeric 0/1 values on an elementwise basis.
        - For each element: try numeric coercion (accept 0/1), booleans, or string tokens.
        - positive_tokens and negative_tokens are sets of tokens (strings) indicating positive/negative values.
        - Token matching uses alphanumeric word extraction and substring fallback to be robust.
        - Unrecognized or missing values return NaN.
        """
        if col_series is None:
            return pd.Series(np.nan, index=df.index, dtype=float)

        pos_tokens = {str(t).strip().lower() for t in (positive_tokens or set())}
        neg_tokens = {str(t).strip().lower() for t in (negative_tokens or set())}

        def map_value(x):
            # Missing / None
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return np.nan

            # Direct boolean
            if isinstance(x, (bool, np.bool_)):
                return 1.0 if bool(x) else 0.0

            # Numeric-like (including numeric strings)
            try:
                num = float(x)
                # Accept exact 0 or 1
                if num == 1.0:
                    return 1.0
                if num == 0.0:
                    return 0.0
                # If numeric but not 0/1, treat as unknown
            except Exception:
                pass

            # Work with string tokens
            s = str(x).strip().lower()
            if s == "" or s in {"none", "nan", "null"}:
                return np.nan

            # Extract words (alphanumeric sequences) for robust matching
            words = set(re.findall(r"[a-z0-9]+", s))

            # Check explicit token sets
            if words & pos_tokens:
                return 1.0
            if words & neg_tokens:
                return 0.0

            # Fallback: substring matching for common roots
            for t in pos_tokens:
                if t in s:
                    return 1.0
            for t in neg_tokens:
                if t in s:
                    return 0.0

            # Common boolean-like words
            if any(w in words for w in {"true", "t", "yes", "y", "1"}):
                return 1.0
            if any(w in words for w in {"false", "f", "no", "n", "0"}):
                return 0.0

            return np.nan

        mapped = col_series.map(map_value)
        # Ensure float dtype
        return pd.Series(mapped.astype(float), index=col_series.index)

    # List of expected raw columns; coerce straightforward numeric raw columns first
    raw_numeric_cols = [
        'feature1', 'feature4', 'feature7', 'feature8', 'feature10', 'feature12'
    ]
    for c in raw_numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Independent variable: Female (1 if female, 0 if male) from feature2
    if 'feature2' in df.columns:
        df['Female'] = parse_binary(
            df['feature2'],
            positive_tokens={'female', 'f', 'woman', 'w', '1', 'yes', 'y', 'true'},
            negative_tokens={'male', 'm', 'man', '0', 'no', 'n', 'false'}
        )
    else:
        df['Female'] = pd.Series(np.nan, index=df.index, dtype=float)

    # Dependent variable: Approved
    # Prefer feature14 (accepted) when present; otherwise derive as 1 - feature11 (denied)
    approved = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature14' in df.columns:
        approved = parse_binary(
            df['feature14'],
            positive_tokens={'accepted', 'accept', 'approved', '1', 'yes', 'y', 'true'},
            negative_tokens={'denied', 'rejected', 'reject', '0', 'no', 'n', 'false'}
        )

    if 'feature11' in df.columns:
        denied_parsed = parse_binary(
            df['feature11'],
            positive_tokens={'denied', 'rejected', 'reject', '1', 'yes', 'y', 'true'},
            negative_tokens={'accepted', 'approved', 'accept', '0', 'no', 'n', 'false'}
        )
        # Fill where approved is missing and denied_parsed is valid
        mask_fill = approved.isna() & denied_parsed.notna()
        if mask_fill.any():
            approved.loc[mask_fill] = 1.0 - denied_parsed.loc[mask_fill]

    # More aggressive vectorized fallback parsing to maximize parsed approvals
    def vectorized_interpret(series: pd.Series, invert_denied: bool = False) -> pd.Series:
        """
        Interpret a series vectorized-ly: look for approval-like substrings, denial-like substrings,
        booleans, and numeric 0/1. If invert_denied is True, interpret numeric 1 as denied (so approved = 0).
        """
        if series is None:
            return pd.Series(np.nan, index=df.index, dtype=float)

        s = series.astype(object).where(series.notna(), None)

        out = pd.Series(np.nan, index=series.index, dtype=float)

        # Numeric parse
        numerics = pd.to_numeric(series, errors='coerce')
        num_mask = numerics.notna()
        if num_mask.any():
            nums = numerics.loc[num_mask]
            if invert_denied:
                out.loc[num_mask & (nums == 1.0)] = 0.0
                out.loc[num_mask & (nums == 0.0)] = 1.0
            else:
                out.loc[num_mask & (nums == 1.0)] = 1.0
                out.loc[num_mask & (nums == 0.0)] = 0.0

        # String inspection for the rest
        def interpret_str(x):
            if x is None:
                return np.nan
            sx = str(x).strip().lower()
            if sx == "" or sx in {"none", "nan", "null"}:
                return np.nan
            # approval indicators
            if re.search(r"\b(accept|approved|approval)\b", sx) or 'accept' in sx or 'approve' in sx:
                return 1.0 if not invert_denied else 1.0  # invert_denied only affects numeric semantics
            if re.search(r"\b(den(y|ied)|reject|rejected)\b", sx) or 'denied' in sx or 'reject' in sx:
                return 0.0 if not invert_denied else 0.0
            # yes/no tokens
            if re.search(r"\b(yes|y)\b", sx):
                return 1.0
            if re.search(r"\b(no|n)\b", sx):
                return 0.0
            # fall back to presence of digit 1 or 0 in text
            if re.search(r"\b1\b", sx):
                return 1.0 if not invert_denied else 0.0
            if re.search(r"\b0\b", sx):
                return 0.0 if not invert_denied else 1.0
            return np.nan

        # Only interpret string rows that remain NaN
        str_mask = out.isna()
        if str_mask.any():
            interpreted = series.loc[str_mask].map(interpret_str)
            out.loc[str_mask] = interpreted

        return out.astype(float)

    # If after initial parsing we have no parsed approvals, try more aggressive parsing
    if approved.notna().sum() == 0:
        if 'feature14' in df.columns:
            tried = vectorized_interpret(df['feature14'], invert_denied=False)
            if tried.notna().sum() > 0:
                approved = tried

    if approved.notna().sum() == 0 and 'feature11' in df.columns:
        # feature11 indicates denied when 1, so invert numeric semantics (1 -> denied -> approved=0)
        tried2 = vectorized_interpret(df['feature11'], invert_denied=True)
        if tried2.notna().sum() > 0:
            approved = tried2

    df['Approved'] = approved

    # Control (binary) variables: use robust parsing; missing -> NaN
    if 'feature3' in df.columns:
        df['Black'] = parse_binary(
            df['feature3'],
            positive_tokens={'black', 'african', 'african-american', 'african american', '1', 'yes', 'y'},
            negative_tokens={'white', 'asian', 'hispanic', '0', 'no', 'n'}
        )
    else:
        df['Black'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature5' in df.columns:
        df['SelfEmployed'] = parse_binary(
            df['feature5'],
            positive_tokens={'self-employed', 'selfemployed', 'self employed', 'self', '1', 'yes', 'y'},
            negative_tokens={'employee', 'not self-employed', 'not self employed', '0', 'no', 'n'}
        )
    else:
        df['SelfEmployed'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature6' in df.columns:
        df['Married'] = parse_binary(
            df['feature6'],
            positive_tokens={'married', 'm', '1', 'yes', 'y', 'true'},
            negative_tokens={'single', 's', '0', 'no', 'n', 'false'}
        )
    else:
        df['Married'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature9' in df.columns:
        df['BadCreditHistory'] = parse_binary(
            df['feature9'],
            positive_tokens={'bad', 'badcredit', 'bad credit', '1', 'yes', 'y', 'true'},
            negative_tokens={'good', '0', 'no', 'n', 'false'}
        )
    else:
        df['BadCreditHistory'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature13' in df.columns:
        df['PMI_Denied'] = parse_binary(
            df['feature13'],
            positive_tokens={'denied', 'pmi denied', '1', 'yes', 'y', 'true', 'rejected'},
            negative_tokens={'approved', 'not denied', '0', 'no', 'n', 'false'}
        )
    else:
        df['PMI_Denied'] = pd.Series(np.nan, index=df.index, dtype=float)

    # Continuous controls (already attempted numeric coercion for some above)
    if 'feature7' in df.columns:
        df['MortgageCreditScore'] = pd.to_numeric(df['feature7'], errors='coerce')
    else:
        df['MortgageCreditScore'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature8' in df.columns:
        df['ConsumerCreditScore'] = pd.to_numeric(df['feature8'], errors='coerce')
    else:
        df['ConsumerCreditScore'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature4' in df.columns:
        df['HousingExpenseRatio'] = pd.to_numeric(df['feature4'], errors='coerce')
    else:
        df['HousingExpenseRatio'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature10' in df.columns:
        df['DebtToIncomeRatio'] = pd.to_numeric(df['feature10'], errors='coerce')
    else:
        df['DebtToIncomeRatio'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature12' in df.columns:
        df['LoanToValue'] = pd.to_numeric(df['feature12'], errors='coerce')
    else:
        df['LoanToValue'] = pd.Series(np.nan, index=df.index, dtype=float)

    if 'feature1' in df.columns:
        df['LoanAmount'] = pd.to_numeric(df['feature1'], errors='coerce')
    else:
        df['LoanAmount'] = pd.Series(np.nan, index=df.index, dtype=float)

    # Standardize continuous controls (z-score). Use population std (ddof=0) to avoid small-sample ddof issues.
    cont_cols = [
        'MortgageCreditScore', 'ConsumerCreditScore', 'HousingExpenseRatio',
        'DebtToIncomeRatio', 'LoanToValue', 'LoanAmount'
    ]
    for c in cont_cols:
        series = df[c].astype(float)
        if series.notna().any():
            mean = series.mean()
            std = series.std(ddof=0)
            if pd.isna(std) or std == 0:
                df['z_' + c] = pd.Series(0.0, index=df.index)
            else:
                df['z_' + c] = (series - mean) / std
        else:
            # No valid observations for this continuous variable; set standardized column to NaN for now
            df['z_' + c] = pd.Series(np.nan, index=df.index)

    # Required outcome column that must be present for modeling
    required_outcome = ['Approved']

    # Drop rows with missing outcome (these are required for the regression).
    # Only drop if there is at least one non-missing Approved value in the dataframe.
    if df['Approved'].notna().any():
        df = df.dropna(subset=required_outcome)
    else:
        # If no Approved values could be parsed at all, keep rows but leave Approved as NaN.
        # The model() function will handle the no-observation case gracefully.
        pass

    # For standardized continuous variables, impute missing z-scores with 0.0 (the standardized mean)
    z_cols = ['z_MortgageCreditScore', 'z_ConsumerCreditScore', 'z_HousingExpenseRatio',
              'z_DebtToIncomeRatio', 'z_LoanToValue', 'z_LoanAmount']
    for z in z_cols:
        if z not in df.columns:
            df[z] = pd.Series(0.0, index=df.index)
        else:
            df[z] = pd.to_numeric(df[z], errors='coerce')
            df[z] = df[z].fillna(0.0)

    # Ensure binary columns are actual integers (0/1) as numpy dtypes.
    # For binary controls including Female, replace remaining NaN with 0 to preserve observations.
    binary_columns = ['Approved', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMI_Denied']
    for b in binary_columns:
        if b not in df.columns:
            df[b] = pd.Series(np.nan, index=df.index, dtype=float)
        df[b] = pd.to_numeric(df[b], errors='coerce')
        # For outcome we have already dropped rows with NaN if any parsed; for other binaries fill NaN with 0
        if b == 'Approved':
            if df[b].isna().any():
                # If still NaN for some rows, leave them as NaN (cannot impute outcome reliably).
                # We will not coerce these to 0 here to avoid introducing bias; model() will check for valid observations.
                pass
        else:
            df[b] = df[b].fillna(0.0)
        # For non-NaN values, cast to integer (0/1) where possible
        # Protect against unexpected non-binary values by clipping to 0/1 after rounding
        # Preserve NaN if present
        df[b] = df[b].where(df[b].isna(), other=df[b].round().clip(0, 1))
        # For columns that are entirely NaN, keep as float NaN; otherwise cast to int
        if df[b].notna().any():
            df[b] = df[b].fillna(0).astype(int)
        else:
            df[b] = df[b].astype(float)

    # Ensure z- columns are float64
    for z in z_cols:
        df[z] = pd.to_numeric(df[z], errors='coerce').astype(float)

    # Return only the columns needed for modeling (keeps transformed columns documented above)
    final_cols = ['Approved', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMI_Denied',
                  'z_MortgageCreditScore', 'z_ConsumerCreditScore', 'z_HousingExpenseRatio',
                  'z_DebtToIncomeRatio', 'z_LoanToValue', 'z_LoanAmount']
    # Ensure all final columns exist
    for col in final_cols:
        if col not in df.columns:
            if col.startswith('z_'):
                df[col] = pd.Series(0.0, index=df.index)
            else:
                df[col] = pd.Series(0, index=df.index, dtype=int)

    return df[final_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (logit) to estimate the effect of applicant gender on mortgage approval
    controlling for applicant and loan characteristics.

    Model: logit( P(Approved=1) ) = beta0 + beta1 * Female + beta2 * Black + beta3 * SelfEmployed + ... + eps

    Returns the fitted statsmodels results object (LogitResults). If there are no observations,
    returns None.
    """
    # Copy to avoid side-effects
    data = df.copy()

    # Predictor columns (as constructed by transform)
    X_cols = [
        'Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMI_Denied',
        'z_MortgageCreditScore', 'z_ConsumerCreditScore', 'z_HousingExpenseRatio',
        'z_DebtToIncomeRatio', 'z_LoanToValue', 'z_LoanAmount'
    ]

    # Ensure all predictors present
    missing = [c for c in X_cols if c not in data.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Ensure correct dtypes (numeric numpy dtypes) to avoid pandas object dtype issues
    X = data[X_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)

    # Add constant using pandas DataFrame so statsmodels retains column info
    X = sm.add_constant(X, has_constant='add')

    y = pd.to_numeric(data['Approved'], errors='coerce')

    # Drop any rows with missing y or missing X (should be minimal because transform cleaned/imputed)
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask].astype(int)

    # Check for empty dataset
    if X.shape[0] == 0 or y.shape[0] == 0:
        # Return None to indicate no model could be fit due to lack of valid observations.
        # This avoids raising an exception and allows callers to handle the no-data case.
        return None

    # Fit logistic regression using statsmodels Logit with pandas objects
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Attach simple design info for easier inspection downstream
    results.model_data = {
        'n_obs': int(results.nobs) if hasattr(results, 'nobs') else int(len(y)),
        'predictor_columns': X_cols,
        'outcome': 'Approved'
    }

    return results