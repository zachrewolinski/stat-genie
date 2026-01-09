from typing import Any, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm


# Helper utilities for robust column discovery and conversion
def _find_column(df: pd.DataFrame, feature_nums: Optional[List[int]] = None, keywords: Optional[List[str]] = None) -> Optional[str]:
    """
    Try to find a suitable source column in df by checking:
      1. feature number patterns (e.g., 'feature2', 'feature_2', 'F2', 'f2')
      2. column names that contain one of the keywords (case-insensitive)
    Returns the first matching column name or None if no candidate found.
    """
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]

    # Try feature number patterns first
    if feature_nums:
        for n in feature_nums:
            candidates = [f'feature{n}', f'feature_{n}', f'f{n}', f'F{n}', f'feat{n}', f'feature-{n}']
            for cand in candidates:
                if cand in cols:
                    return cand
                if cand.lower() in lower_cols:
                    return cols[lower_cols.index(cand.lower())]

    # Then try keyword matching anywhere in column name
    if keywords:
        for kw in keywords:
            if kw is None:
                continue
            kw = kw.lower()
            for i, lc in enumerate(lower_cols):
                if kw in lc:
                    return cols[i]
    return None


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """
    Try to coerce various representations to numeric values (floats).
    Preserves NaNs.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    # Convert boolean to float
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    # If object/string, try parsing numbers first
    coerced = pd.to_numeric(series, errors='coerce')
    if coerced.notna().any():
        return coerced.astype(float)
    # Otherwise map common truthy/falsey strings
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        'yes': 1.0, 'y': 1.0, 'true': 1.0, 't': 1.0, '1': 1.0,
        'no': 0.0, 'n': 0.0, 'false': 0.0, 'f': 0.0, '0': 0.0
    }
    return s.map(mapping).astype(float)


def _to_indicator(series: pd.Series, positive_keywords: Optional[List[str]] = None) -> pd.Series:
    """
    Convert a series to a 0/1 indicator float series.
    - If numeric and appears binary (0/1), use it.
    - If numeric but not binary, treat non-zero as 1.
    - If strings, mark as 1 when any of positive_keywords is contained (case-insensitive),
      otherwise try common yes/no mappings.
    """
    if positive_keywords is None:
        positive_keywords = []

    # If numeric
    if pd.api.types.is_numeric_dtype(series):
        uniq = pd.unique(series[~pd.isna(series)])
        # If values already 0/1 or only one or two unique non-nan values treat accordingly
        try:
            uniq_set = set(uniq)
        except Exception:
            uniq_set = set([float(x) for x in uniq if not pd.isna(x)])
        if uniq_set.issubset({0, 1}) or len(uniq) <= 2:
            return series.fillna(0).astype(float)
        else:
            # Non-zero considered positive
            return (series != 0).astype(float)

    # If boolean
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    # Strings/objects
    s = series.astype(str).str.strip().str.lower()
    # Check positive keywords
    if positive_keywords:
        mask = pd.Series(False, index=s.index)
        for kw in positive_keywords:
            if kw is None:
                continue
            mask = mask | s.str.contains(kw.lower(), na=False)
        # Also allow common yes/true strings
        yes_map = s.isin({'yes', 'y', 'true', 't', '1'})
        mask = mask | yes_map
        return mask.astype(float)

    # Generic yes/no mapping
    mapped = _to_numeric_series(series)
    if mapped.notna().any():
        # if mapping succeeded into numbers, treat non-zero as 1
        return (mapped != 0).astype(float).fillna(0.0)

    # As a last resort, consider non-empty strings as positive
    return (s != '').astype(float)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Prepare a place for final required columns
    # Find columns for Approved: prefer feature14 (accepted) else invert feature11 (denied).
    approved_col = _find_column(df, feature_nums=[14], keywords=['approved', 'accepted', 'accept'])
    denied_col = _find_column(df, feature_nums=[11], keywords=['denied', 'rejected', 'reject', 'deny', 'decline', 'declined'])

    if approved_col is not None:
        # Use approved_col directly
        approved_series = _to_indicator(df[approved_col], positive_keywords=['approved', 'accepted', 'accept', 'yes'])
    elif denied_col is not None:
        # Invert denied indicator: Approved = 1 - denied
        denied_series = _to_indicator(df[denied_col], positive_keywords=['denied', 'rejected', 'reject', 'deny', 'decline'])
        approved_series = (1.0 - denied_series).astype(float)
    else:
        # Try to find any column with 'decision' or 'status' that could indicate approval
        alt_col = _find_column(df, keywords=['decision', 'status', 'outcome', 'result'])
        if alt_col is not None:
            # Try to map strings like 'approved'/'denied'
            s = df[alt_col].astype(str).str.strip().str.lower()
            # approved if contains 'approve' or 'accept'
            approved_series = (s.str.contains('approve', na=False) | s.str.contains('accept', na=False) | s.isin({'1', 'yes', 'y', 'true'})).astype(float)
        else:
            raise KeyError("Could not find a column for Approved (neither feature14 nor feature11 nor common alternatives exist). Available columns: " + ", ".join(df.columns))

    df['Approved'] = approved_series

    # Independent variable: Female (feature2 or alternatives)
    female_col = _find_column(df, feature_nums=[2], keywords=['female', 'sex', 'gender', 'woman'])
    if female_col is None:
        # try generic patterns where column name is 'f' or 'is_female'
        female_col = _find_column(df, keywords=['is_female', 'female_indicator', 'isfemale', 'f_', ' f ', 'female'])
    if female_col is None:
        raise KeyError("Could not find a column to map to 'Female' (tried feature2 and common alternatives). Available columns: " + ", ".join(df.columns))
    df['Female'] = _to_indicator(df[female_col], positive_keywords=['female', 'woman', 'f', 'woman'])

    # Controls: map columns to clear names using feature numbers or reasonable keywords
    mappings = {
        'Black': {'feature_nums': [3], 'keywords': ['black', 'race', 'african']},
        'HousingExpenseRatio': {'feature_nums': [4], 'keywords': ['housingexpense', 'housing_expense', 'housing', 'expense', 'housingratio', 'housing_expense_ratio']},
        'SelfEmployed': {'feature_nums': [5], 'keywords': ['selfemploy', 'self_employed', 'self-employed', 'selfemp', 'self']},
        'Married': {'feature_nums': [6], 'keywords': ['married', 'marital', 'marital_status']},
        'MortgageScore': {'feature_nums': [7], 'keywords': ['mortgage_score', 'mortgage', 'mortgage_credit', 'mortgagescore', 'mortgage_credit']},
        'ConsumerScore': {'feature_nums': [8], 'keywords': ['consumer_score', 'consumer', 'credit_score', 'consumerscore', 'consumer_credit']},
        'BadCreditHistory': {'feature_nums': [9], 'keywords': ['badcredit', 'bad_credit', 'bad_credit_history', 'bankruptcy', 'delinquent', 'bad_history', 'bad history', 'bad_history']},
        # Expanded keywords for DebtToIncomeRatio to capture common alternate names (e.g., PI_ratio, payment-to-income, pti)
        'DebtToIncomeRatio': {'feature_nums': [10], 'keywords': ['debttoincome', 'debt_to_income', 'dti', 'debt_income', 'debt_to_income_ratio', 'pi', 'pi_ratio', 'pi ratio', 'payment_to_income', 'payment_income', 'payment_to_income_ratio', 'pti', 'pti_ratio', 'payment_income_ratio', 'pi_ratio']},
        'LoanToValueRatio': {'feature_nums': [12], 'keywords': ['loan_to_value', 'ltv', 'loanvalue', 'loan_to_value_ratio', 'loan to value', 'loan_to_value']},
        'PMI_Denied': {'feature_nums': [13], 'keywords': ['pmi', 'pmi_denied', 'private_mortgage_insurance', 'pmi_denial', 'pmi_reject', 'pmi_rejected', 'denied_pmi', 'denied_pmi']}
    }

    # Iterate and assign
    for final_col, tryinfo in mappings.items():
        col = _find_column(df, feature_nums=tryinfo.get('feature_nums'), keywords=tryinfo.get('keywords'))
        if col is None:
            raise KeyError(f"Could not find a column to map to '{final_col}'. Tried feature numbers {tryinfo.get('feature_nums')} and keywords {tryinfo.get('keywords')}. Available columns: " + ", ".join(df.columns))
        # For indicator variables, use indicator conversion; for continuous keep numeric
        if final_col in ['Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMI_Denied']:
            # Black may be categorical string; treat as indicator if contains 'black'
            positive_keywords = []
            if final_col == 'Black':
                positive_keywords = ['black', 'african']
            elif final_col == 'SelfEmployed':
                positive_keywords = ['self']
            elif final_col == 'Married':
                positive_keywords = ['married']
            elif final_col == 'BadCreditHistory':
                positive_keywords = ['bad', 'bankrupt', 'delinq', 'delinquent', 'bad_history']
            elif final_col == 'PMI_Denied':
                positive_keywords = ['denied', 'rejected', 'deny', 'pmi']
            df[final_col] = _to_indicator(df[col], positive_keywords=positive_keywords)
        else:
            # Continuous: coerce to numeric
            numeric = _to_numeric_series(df[col])
            df[final_col] = numeric

    # Required columns for the model
    required_cols = [
        'Approved', 'Female', 'Black', 'SelfEmployed', 'Married',
        'MortgageScore', 'ConsumerScore', 'BadCreditHistory',
        'HousingExpenseRatio', 'DebtToIncomeRatio', 'LoanToValueRatio', 'PMI_Denied'
    ]

    # Drop rows that are missing any required variable
    df = df.dropna(subset=required_cols).copy()

    # Ensure binary columns are numeric 0/1 floats (for listed binary columns)
    for bcol in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCreditHistory', 'PMI_Denied', 'Approved']:
        if bcol in df.columns:
            # Already indicator in most cases, but coerce to 0/1
            df[bcol] = (df[bcol] != 0).astype(float)

    # Standardize continuous predictors (z-score). Use population std (ddof=0) for stability.
    cont_cols = ['MortgageScore', 'ConsumerScore', 'HousingExpenseRatio', 'DebtToIncomeRatio', 'LoanToValueRatio']
    for c in cont_cols:
        if c not in df.columns:
            # Shouldn't happen due to previous checks, but guard anyway
            df['z_' + c] = np.nan
            continue
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        if std == 0 or np.isnan(std):
            df['z_' + c] = 0.0
        else:
            df['z_' + c] = (df[c] - mean) / std

    # Final dataframe contains all model columns (must keep exact names)
    final_cols = ['Approved', 'Female', 'Black', 'SelfEmployed', 'Married',
                  'z_MortgageScore', 'z_ConsumerScore', 'BadCreditHistory',
                  'z_HousingExpenseRatio', 'z_DebtToIncomeRatio', 'z_LoanToValueRatio', 'PMI_Denied']

    # Ensure all required final columns exist
    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        raise KeyError(f"After transformation the following required final columns are missing: {missing}")

    # Return df (may contain additional columns but must include final ones)
    return df


def model(df: pd.DataFrame) -> Any:
    df = df.copy()

    # Outcome and predictors
    if 'Approved' not in df.columns:
        raise KeyError("Input dataframe to model() must contain 'Approved' column.")
    y = df['Approved'].astype(float)

    needed_X = [
        'Female', 'Black', 'SelfEmployed', 'Married',
        'BadCreditHistory', 'PMI_Denied',
        'z_MortgageScore', 'z_ConsumerScore',
        'z_HousingExpenseRatio', 'z_DebtToIncomeRatio', 'z_LoanToValueRatio'
    ]
    missing = [c for c in needed_X if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe to model() is missing required predictor columns: {missing}")

    X = df[needed_X].astype(float)

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (binary outcome) using statsmodels Logit
    logit = sm.Logit(y, X)
    res = logit.fit(disp=False)

    # Compute robust (heteroskedasticity-consistent) covariance estimates if supported by this statsmodels version.
    # Some statsmodels versions provide get_robustcov_results on result objects; others do not.
    if hasattr(res, 'get_robustcov_results'):
        try:
            res_robust = res.get_robustcov_results(cov_type='HC3')
        except Exception:
            res_robust = res
    else:
        # Fallback: keep original results object when robust wrapper is unavailable.
        res_robust = res

    # Compute average marginal effects. Try using the robust results object first; if that fails, fall back.
    margeff = None
    try:
        # Many versions implement get_margeff on results; try robust results first (if it's a wrapper)
        margeff = res_robust.get_margeff(at='overall', method='dydx')
    except Exception:
        try:
            margeff = res.get_margeff(at='overall', method='dydx')
        except Exception:
            margeff = None

    return {
        'model_results_robust': res_robust,
        'marginal_effects': margeff
    }