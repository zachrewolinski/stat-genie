from typing import Any
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling.
    - Constructs a single approval column 'approved' (1=approved,0=denied).
      Preference order: existing 'approved' column, then 'feature14' (1=accepted,0=denied),
      then 'feature11' (1=denied,0=accepted) which is inverted.
    - Maps relevant features to clear column names.
    - Standardizes continuous variables used as controls (z-scores): amount, debt_to_income, loan_to_value.
    - Does NOT drop rows here; the model function will handle dropping observations as needed.

    Final columns created and returned (minimum set used by the model):
      - approved, female, black, self_employed, married, bad_credit,
        mortgage_score, consumer_score, amount_z, debt_to_income_z, loan_to_value_z, pmi_denied
    """
    df = df.copy()

    # Build case-insensitive and normalized column name maps
    col_map = {col.lower(): col for col in df.columns}

    def _normalize(name: str) -> str:
        # remove non-alphanumeric characters and lowercase for robust matching
        return re.sub(r'[^a-z0-9]', '', name.lower())

    norm_map = {}
    for col in df.columns:
        norm = _normalize(col)
        # keep first occurrence if collisions
        if norm not in norm_map:
            norm_map[norm] = col

    def _find_col(name: str):
        """
        Find a column in df by:
          1) exact case-insensitive match,
          2) normalized match (strip non-alphanum, lower),
          3) normalized substring match.
        Returns actual column name or None.
        """
        if name is None:
            return None
        key = name.lower()
        if key in col_map:
            return col_map[key]
        norm = _normalize(name)
        if norm in norm_map:
            return norm_map[norm]
        # substring match on normalized names
        for n, actual in norm_map.items():
            if norm in n:
                return actual
        return None

    def _map_binary_from_series(s: pd.Series) -> pd.Series:
        """
        Try to map a series to binary 1/0. Attempts numeric coercion first;
        only accepts numeric coercion if the resulting non-null values are
        exclusively 0/1. Otherwise attempts to map common string labels to 1.0/0.0.
        Returns a float Series with values 0.0/1.0 or NaN where mapping failed.
        """
        # Try numeric coercion
        numeric = pd.to_numeric(s, errors='coerce')
        non_null = numeric.dropna()
        if not non_null.empty:
            unique_vals = set(np.unique(non_null.values))
            # Accept numeric mapping only if values are binary 0/1
            if unique_vals.issubset({0, 1, 0.0, 1.0}):
                return numeric.astype(float)

        # Try mapping common textual labels
        s_str = s.astype(str).str.strip().str.lower()
        mapping = {
            'approved': 1.0, 'accepted': 1.0, 'accept': 1.0, 'yes': 1.0, 'y': 1.0, 'true': 1.0, '1': 1.0,
            'denied': 0.0, 'rejected': 0.0, 'reject': 0.0, 'no': 0.0, 'n': 0.0, 'false': 0.0, '0': 0.0,
            'female': 1.0, 'f': 1.0, 'male': 0.0, 'm': 0.0
        }
        mapped = s_str.map(mapping)

        return mapped.astype(float)

    # 1) Approval indicator: prefer existing 'approved', then feature14, then inverted feature11
    approved_series = None

    approved_col = _find_col('approved')
    feature14_col = _find_col('feature14')
    feature11_col = _find_col('feature11')

    if approved_col is not None:
        approved_series = _map_binary_from_series(df[approved_col])
    elif feature14_col is not None:
        approved_series = _map_binary_from_series(df[feature14_col])
    elif feature11_col is not None:
        # feature11: 1 if denied, 0 if accepted -> invert to get approved indicator
        tmp = _map_binary_from_series(df[feature11_col])
        approved_series = tmp.apply(lambda x: 1.0 - x if pd.notna(x) else np.nan)
    else:
        # As a last fallback, try to find any column whose name contains 'approve' (case-insensitive)
        found = None
        for col in df.columns:
            if 'approve' in col.lower():
                found = col
                break
        if found is not None:
            approved_series = _map_binary_from_series(df[found])

    # Additional robust fallback: try columns with '14' or that look binary (0/1) and pick the most complete one
    if approved_series is None:
        # First try normalized-name candidates that contain '14' (handles things like 'feature_14', 'f14', etc.)
        candidate = None
        best_count = -1
        for n, actual in norm_map.items():
            if '14' in n or 'feature14' in n or n.endswith('f14') or n.endswith('feat14'):
                mapped = _map_binary_from_series(df[actual])
                non_null_count = int(mapped.notna().sum())
                non_null = mapped.dropna()
                uniq = set(np.unique(non_null.values)) if not non_null.empty else set()
                if non_null_count > 0 and uniq.issubset({0.0, 1.0}):
                    if non_null_count > best_count:
                        best_count = non_null_count
                        candidate = mapped
        if candidate is not None:
            approved_series = candidate

    if approved_series is None:
        # Generic auto-detection: pick the column (excluding obvious predictors) that maps cleanly to binary 0/1
        candidate = None
        best_count = -1
        excluded_cols = {feature14_col, feature11_col, approved_col}
        # also exclude feature2 (gender), feature1 (amount) to avoid mis-identifying predictors as approval
        excluded_cols.update({_find_col('feature2'), _find_col('feature1')})
        for col in df.columns:
            if col in excluded_cols:
                continue
            mapped = _map_binary_from_series(df[col])
            non_null = mapped.dropna()
            non_null_count = int(non_null.shape[0])
            if non_null_count == 0:
                continue
            uniq = set(np.unique(non_null.values))
            if uniq.issubset({0.0, 1.0}):
                # prefer column with most non-missing entries
                if non_null_count > best_count:
                    best_count = non_null_count
                    candidate = mapped
        if candidate is not None:
            approved_series = candidate

    if approved_series is None:
        # If we still couldn't derive an approval column, attempt to find variants like 'approval' or 'status'
        found = None
        for col in df.columns:
            lname = col.lower()
            if any(k in lname for k in ['approval', 'status', 'decision']):
                found = col
                break
        if found is not None:
            approved_series = _map_binary_from_series(df[found])

    if approved_series is None:
        raise ValueError('Neither feature14 nor feature11 nor approved column found to derive approval outcome')

    df['approved'] = approved_series.astype(float)

    # Map main independent variable (gender) - robust search for likely gender columns
    # Try several candidate source names in order of preference.
    gender_candidate_names = [
        'feature2', 'sex', 'gender', 'applicant_sex', 'applicantsex', 'gendercode', 'sexcode', 'applicant_gender',
        'applicantgender', 'gender_id', 'gender_id', 'f/m', 'f_m', 'male', 'female'
    ]
    feature2_col = None
    for cand in gender_candidate_names:
        col = _find_col(cand)
        if col is not None:
            feature2_col = col
            break

    # If still not found, try any column whose name contains 'gender' or 'sex'
    if feature2_col is None:
        for col in df.columns:
            lname = col.lower()
            if 'gender' in lname or 'sex' in lname:
                feature2_col = col
                break

    if feature2_col is not None:
        # Use binary mapper to handle textual gender encodings as well as numeric 0/1
        df['female'] = _map_binary_from_series(df[feature2_col]).astype(float)
    else:
        raise ValueError('feature2 (gender) is required but not present')

    # Controls mapping (binary flags and numeric predictors) - robust mapping
    def _map_optional_numeric(target_col_name: str):
        col = _find_col(target_col_name)
        if col is not None:
            return pd.to_numeric(df[col], errors='coerce')
        else:
            # return a Series of NaNs matching df length to allow downstream code to handle missingness uniformly
            return pd.Series(np.nan, index=df.index, dtype=float)

    df['black'] = _map_optional_numeric('feature3')
    df['self_employed'] = _map_optional_numeric('feature5')
    df['married'] = _map_optional_numeric('feature6')
    df['mortgage_score'] = _map_optional_numeric('feature7')
    df['consumer_score'] = _map_optional_numeric('feature8')
    df['bad_credit'] = _map_optional_numeric('feature9')
    df['debt_to_income'] = _map_optional_numeric('feature10')
    df['pmi_denied'] = _map_optional_numeric('feature13')
    df['loan_to_value'] = _map_optional_numeric('feature12')
    df['amount'] = _map_optional_numeric('feature1')

    # Standardize continuous predictors (z-score). Using sample std (ddof=1) for scaling.
    for col in ['amount', 'debt_to_income', 'loan_to_value']:
        zcol = col + '_z'
        mean = df[col].mean()
        std = df[col].std(ddof=1)
        if std == 0 or np.isnan(std):
            # If no variation (or entirely NaN), create zero column to avoid division by zero.
            df[zcol] = 0.0
        else:
            df[zcol] = (df[col] - mean) / std

    # Keep only the columns required for modeling (makes the downstream model code explicit)
    model_ready_cols = [
        'approved', 'female', 'black', 'self_employed', 'married', 'bad_credit',
        'mortgage_score', 'consumer_score', 'amount_z', 'debt_to_income_z', 'loan_to_value_z', 'pmi_denied'
    ]

    # Ensure the final columns exist in the dataframe (if any are missing, create as NaN so caller gets a consistent schema)
    for col in model_ready_cols:
        if col not in df.columns:
            df[col] = np.nan

    # NOTE: Do not drop rows here. Let the model function handle dropping rows with missing data
    # so that callers can inspect transformed data if desired.

    # Reorder and return only the model-ready columns
    return df[model_ready_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial) model to estimate the effect of gender on mortgage approval,
    controlling for credit/financial and demographic covariates.

    Model specification (logit):
      approved ~ female + black + self_employed + married + bad_credit
                 + mortgage_score + consumer_score + amount_z + debt_to_income_z
                 + loan_to_value_z + pmi_denied

    Returns the fitted statsmodels results object (LogitResults).
    """
    df = df.copy()

    # Ensure columns exist
    X_cols = [
        'female', 'black', 'self_employed', 'married', 'bad_credit',
        'mortgage_score', 'consumer_score', 'amount_z', 'debt_to_income_z', 'loan_to_value_z', 'pmi_denied'
    ]

    # Check required columns are present
    missing_cols = [c for c in X_cols + ['approved'] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required model columns: {missing_cols}")

    # Prepare design matrix: coerce to numeric
    X = df[X_cols].apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(df['approved'], errors='coerce')

    # Require non-missing outcome and primary independent variable (female).
    data = pd.concat([y, X], axis=1)
    data = data.dropna(subset=['approved', 'female'])
    if data.shape[0] == 0:
        raise ValueError("No observations available after requiring non-missing 'approved' and 'female'.")

    # Impute missing control variables in a conservative, transparent way:
    # - For binary-like controls (values only 0/1 among non-missing) use mode
    # - For continuous controls use mean
    # - If a column is entirely missing, fill with 0.0
    for col in X_cols:
        if col == 'female':
            # female is required and should be non-missing at this point
            continue
        series = data[col]
        if series.notna().any():
            non_null = series.dropna()
            unique_vals = set(np.unique(non_null.values))
            # treat as binary if all observed values are 0/1
            if unique_vals.issubset({0, 1, 0.0, 1.0}):
                # use mode; if tie, mode().iloc[0] picks the smallest
                mode = non_null.mode()
                fill = float(mode.iloc[0]) if not mode.empty else 0.0
                data[col] = series.fillna(fill)
            else:
                mean = non_null.mean()
                fill = mean if not np.isnan(mean) else 0.0
                data[col] = series.fillna(fill)
        else:
            # no observed values at all -> set to 0.0 (neutral baseline)
            data[col] = 0.0

    y = data['approved']
    X = data[X_cols]

    # Add constant term (intercept)
    X = sm.add_constant(X, has_constant='add')

    if X.shape[1] == 0:
        raise ValueError("Exogenous design matrix has zero columns after processing; cannot fit model.")

    # Fit logistic regression using statsmodels Logit. Use robust (HC1) standard errors for inference in the summary if desired.
    logit = sm.Logit(y, X)
    try:
        res = logit.fit(disp=False)
    except Exception:
        # If the default solver fails (perfect separation, convergence), try a different optimizer and regularization fallback.
        res = logit.fit(method='bfgs', maxiter=1000, disp=False)

    # Print a short summary to the console (optional) and return the results object
    print(res.summary())

    return res