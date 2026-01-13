from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the Boston mortgage dataset for a logistic regression analysis of Gender on Approval.

    Outputs (columns used in modeling):
      - Approval: 1 = accepted, 0 = denied
      - Gender: 1 = female, 0 = male
      - Black: 1 = applicant is Black, 0 otherwise
      - Gender_Black: interaction term Gender * Black
      - MortgageScore_z, ConsumerScore_z, DebtToIncome_z, LoanToValue_z, HousingExpenseRatio_z, Income_z: standardized continuous controls
      - BadCredit, SelfEmployed, Married: binary controls
    """
    df = df.copy()

    # Helper: find first column name containing any of the provided keywords
    def _find_col(keywords):
        if not keywords:
            return None
        kws = [k.lower() for k in keywords]
        for col in df.columns:
            col_l = col.lower()
            for k in kws:
                if k in col_l:
                    return col
        return None

    # Helper: find all column names containing any of the provided keywords
    def _find_all_cols(keywords):
        if not keywords:
            return []
        kws = [k.lower() for k in keywords]
        out = []
        for col in df.columns:
            col_l = col.lower()
            if any(k in col_l for k in kws):
                out.append(col)
        return out

    # Helper: safely get a numeric series from df or return a NaN series if missing
    def _safe_numeric(col_name: str) -> pd.Series:
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors='coerce')
        else:
            return pd.Series(np.nan, index=df.index, dtype=float)

    # Helper: map common textual/numeric encodings to binary 0/1, returning float series with NaN when ambiguous
    def _map_binary_series(series: pd.Series, positive_tokens=None, negative_tokens=None) -> pd.Series:
        # Work on a copy
        s = series.copy()
        # If numeric dtype, handle directly
        if pd.api.types.is_numeric_dtype(s):
            uniq = pd.unique(s.dropna())
            if set(uniq).issubset({0, 1, 0.0, 1.0}):
                return s.astype(float)
            out_num = s.apply(lambda x: 1.0 if pd.notna(x) and x != 0 else (0.0 if x == 0 else np.nan)).astype(float)
            return out_num

        # For non-numeric, normalize strings
        s_str = s.astype(str).str.strip().str.lower()
        out = pd.Series(np.nan, index=s.index, dtype=float)

        pos = set([t.lower() for t in (positive_tokens or [])])
        neg = set([t.lower() for t in (negative_tokens or [])])

        for idx in s_str.index:
            raw = s.loc[idx]
            val = s_str.loc[idx]
            if pd.isna(raw) or val in ('nan', 'none', ''):
                out.loc[idx] = np.nan
                continue
            if val in pos:
                out.loc[idx] = 1.0
            elif val in neg:
                out.loc[idx] = 0.0
            else:
                try:
                    num = float(val)
                    out.loc[idx] = 1.0 if num != 0 else 0.0
                except Exception:
                    out.loc[idx] = np.nan
        return out

    # Specific mappers for conceptual variables
    def _map_gender(col_name: str) -> pd.Series:
        if col_name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        s = df[col_name]
        pos = {'female', 'f', 'woman', 'women', 'female.', 'fem', 'girl'}
        neg = {'male', 'm', 'man', 'men', 'male.', 'masc', 'boy'}
        return _map_binary_series(s, positive_tokens=pos, negative_tokens=neg)

    def _map_black(col_name: str) -> pd.Series:
        if col_name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        s = df[col_name]
        pos = {'black', 'african american', 'african-american', 'african_american', 'aa'}
        neg = {'white', 'asian', 'hispanic', 'latino', 'other', 'not black', 'non-black', 'non black'}
        return _map_binary_series(s, positive_tokens=pos, negative_tokens=neg)

    def _map_generic_flag(col_name: str) -> pd.Series:
        if col_name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        s = df[col_name]
        pos = {'yes', 'y', '1', 'true', 't'}
        neg = {'no', 'n', '0', 'false', 'f'}
        return _map_binary_series(s, positive_tokens=pos, negative_tokens=neg)

    def _map_approval(col_name: str, invert: bool = False) -> pd.Series:
        # invert=True means input is "denied" indicator (1 = denied)
        if col_name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        s = df[col_name]
        # Numeric preferred
        if pd.api.types.is_numeric_dtype(s):
            uniq = pd.unique(s.dropna())
            if set(uniq).issubset({0, 1, 0.0, 1.0}):
                out = s.astype(float)
                if invert:
                    out = (1.0 - out)
                return out
            out = s.apply(lambda x: 1.0 if pd.notna(x) and x != 0 else (0.0 if x == 0 else np.nan)).astype(float)
            if invert:
                out = 1.0 - out
            return out

        # String-based mapping
        s_str = s.astype(str).str.strip().str.lower()
        out = pd.Series(np.nan, index=s.index, dtype=float)
        accept_tokens = {'accepted', 'approved', 'approve', 'yes', 'accept', 'a'}
        deny_tokens = {'denied', 'rejected', 'reject', 'no', 'd'}
        for idx in s_str.index:
            val = s_str.loc[idx]
            if val in ('nan', 'none', ''):
                out.loc[idx] = np.nan
                continue
            if val in accept_tokens:
                val_num = 1.0
            elif val in deny_tokens:
                val_num = 0.0
            else:
                try:
                    num = float(val)
                    val_num = 1.0 if num != 0 else 0.0
                except Exception:
                    val_num = np.nan
            out.loc[idx] = (1.0 - val_num) if (invert and pd.notna(val_num)) else val_num
        return out

    # Build a list of approval-like candidate columns, preferring known feature names but also searching
    approval_candidates = []
    if 'feature14' in df.columns:
        approval_candidates.append(('feature14', False))
    if 'feature11' in df.columns:
        approval_candidates.append(('feature11', True))

    # Find any other columns that look like approval/decision/status
    approval_keywords = ['approval', 'approve', 'accepted', 'accepted_flag', 'decision', 'status', 'result', 'accept', 'deny', 'denied', 'rejected', 'reject']
    found_approval_cols = _find_all_cols(approval_keywords)
    for col in found_approval_cols:
        if col not in [c for c, _ in approval_candidates]:
            invert = any(token in col.lower() for token in ['deny', 'denied', 'reject', 'rejected', 'declin'])
            approval_candidates.append((col, invert))

    # Map all candidate approval columns and combine them to produce a single Approval column with maximal information.
    mapped_approvals = {}
    for col_name, invert_flag in approval_candidates:
        try:
            mapped = _map_approval(col_name, invert=invert_flag)
            # Only keep if mapping produced any non-null values
            if mapped.notna().any():
                mapped_approvals[col_name] = mapped
        except Exception:
            # If a mapping fails for some reason, skip that candidate
            continue

    if mapped_approvals:
        mapped_df = pd.DataFrame(mapped_approvals)
        # Combine row-wise:
        # - If any candidate indicates accepted (1), set Approval = 1
        # - Else if any candidate indicates denied (0), set Approval = 0
        # - Else NaN
        approval_series = pd.Series(np.nan, index=df.index, dtype=float)
        # Rows where any mapped == 1
        any_accept = (mapped_df == 1.0).any(axis=1)
        approval_series.loc[any_accept] = 1.0
        # Rows not already set and where any mapped == 0
        not_set = approval_series.isna()
        any_deny = (mapped_df == 0.0).any(axis=1)
        approval_series.loc[not_set & any_deny] = 0.0

        # For rows still missing, attempt to take the candidate with the most non-missing values (fallback)
        still_missing = approval_series.isna()
        if still_missing.any():
            # Choose candidate with most non-missing entries
            counts = {c: mapped_df[c].notna().sum() for c in mapped_df.columns}
            if counts:
                best_col = max(counts, key=lambda k: counts[k])
                approval_series.loc[still_missing] = mapped_df[best_col].loc[still_missing]

        df['Approval'] = approval_series
    else:
        # No candidates found: create an all-NaN Approval column
        df['Approval'] = pd.Series(np.nan, index=df.index, dtype=float)

    # Map Gender: prefer feature2, else search
    gender_col = 'feature2' if 'feature2' in df.columns else _find_col(['gender', 'sex', 'female', 'male'])
    if gender_col is not None:
        df['Gender'] = _map_gender(gender_col)
    else:
        df['Gender'] = pd.Series(np.nan, index=df.index, dtype=float)

    # Map Black: prefer feature3, else search
    black_col = 'feature3' if 'feature3' in df.columns else _find_col(['race', 'ethnic', 'black', 'african'])
    if black_col is not None:
        df['Black'] = _map_black(black_col)
    else:
        df['Black'] = pd.Series(np.nan, index=df.index, dtype=float)

    # Binary controls: prefer feature9, 5, 6; else search by likely names
    badcredit_col = 'feature9' if 'feature9' in df.columns else _find_col(['badcredit', 'bad_credit', 'bad credit', 'badcredithistory', 'history_bad', 'bad_credit_history'])
    selfemp_col = 'feature5' if 'feature5' in df.columns else _find_col(['self', 'selfemploy', 'self_employ', 'self-employed', 'selfemployed'])
    married_col = 'feature6' if 'feature6' in df.columns else _find_col(['married', 'marital', 'marital_status', 'spouse'])

    df['BadCredit'] = _map_generic_flag(badcredit_col) if badcredit_col is not None else pd.Series(np.nan, index=df.index, dtype=float)
    df['SelfEmployed'] = _map_generic_flag(selfemp_col) if selfemp_col is not None else pd.Series(np.nan, index=df.index, dtype=float)
    df['Married'] = _map_generic_flag(married_col) if married_col is not None else pd.Series(np.nan, index=df.index, dtype=float)

    # Continuous controls -- try preferred feature names, otherwise search columns by keyword hints
    cont_map = {
        'Income': ('feature1', ['income', 'annual_income', 'salary']),
        'HousingExpenseRatio': ('feature4', ['housingexpense', 'housing_expense', 'housing', 'housingratio', 'housing_expense_ratio']),
        'MortgageScore': ('feature7', ['mortgage', 'mortgage_score', 'mortgagescore', 'score7']),
        'ConsumerScore': ('feature8', ['consumer', 'consumer_score', 'consumerscore', 'score8']),
        'DebtToIncome': ('feature10', ['debttoincome', 'debt_to_income', 'dti', 'debt-income']),
        'LoanToValue': ('feature12', ['loantovalue', 'loan_to_value', 'ltv']),
    }

    for new_name, (preferred, keywords) in cont_map.items():
        src = preferred if preferred in df.columns else _find_col(keywords)
        if src is None:
            df[new_name] = pd.Series(np.nan, index=df.index, dtype=float)
        else:
            df[new_name] = pd.to_numeric(df[src], errors='coerce')

    # Standardize continuous variables (z-scores). Use population std (ddof=0) for stability.
    for base in [k for k in cont_map.keys()]:
        col_z = base + '_z'
        mean = df[base].mean(skipna=True)
        std = df[base].std(ddof=0, skipna=True)
        if pd.isna(std) or std == 0:
            if df[base].notna().any() and std == 0:
                df[col_z] = 0.0
            else:
                df[col_z] = pd.Series(np.nan, index=df.index, dtype=float)
        else:
            df[col_z] = (df[base] - mean) / std

    # Interaction (explicit) for moderation test
    df['Gender_Black'] = df['Gender'] * df['Black']

    # Replace infinite values produced by any operations with NaN so they are dropped
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Define the columns required for the model and ensure they exist
    required_cols = [
        'Approval', 'Gender', 'Black', 'Gender_Black',
        'MortgageScore_z', 'ConsumerScore_z', 'BadCredit', 'SelfEmployed', 'Married',
        'DebtToIncome_z', 'LoanToValue_z', 'HousingExpenseRatio_z', 'Income_z'
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.Series(np.nan, index=df.index, dtype=float)

    # Only drop rows that have neither Approval nor Gender information at all (be permissive)
    df = df.dropna(subset=['Approval', 'Gender'], how='all')

    # Ensure numeric dtype for modeling for all required columns (coerce non-numeric to NaN)
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression of Approval on Gender with controls. Includes an interaction Gender * Black to test whether the gender effect differs by race.

    Returns the fitted statsmodels Logit result object.
    """
    # Basic checks to provide informative errors rather than low-level numpy errors
    if not isinstance(df, pd.DataFrame):
        raise TypeError("model() expects a pandas DataFrame as input (the output of transform()).")
    if df.shape[0] == 0:
        raise ValueError("No data available for modeling: the dataframe has zero rows after transform().")
    if 'Approval' not in df.columns:
        raise ValueError("Required column 'Approval' not found in dataframe.")
    if 'Gender' not in df.columns:
        raise ValueError("Required column 'Gender' not found in dataframe.")

    # Ensure there is variation in the outcome (exclude NA)
    if df['Approval'].dropna().nunique() < 2:
        raise ValueError("The dependent variable 'Approval' has no variation (constant or all-missing). Cannot fit logistic regression.")

    # Base RHS variables (as conceptualized)
    rhs_vars = [
        'Gender', 'Black', 'Gender_Black',
        'MortgageScore_z', 'ConsumerScore_z', 'BadCredit', 'SelfEmployed', 'Married',
        'DebtToIncome_z', 'LoanToValue_z', 'HousingExpenseRatio_z', 'Income_z'
    ]

    # Determine which RHS variables have any non-missing data anywhere in the dataframe.
    # If a variable is entirely missing, drop it from the formula to avoid requiring complete data on it.
    available_rhs = [v for v in rhs_vars if v in df.columns and df[v].notna().any()]

    # Gender must be available for answering the research question
    if 'Gender' not in available_rhs:
        raise ValueError("Required predictor 'Gender' has no non-missing values. Cannot fit the model.")

    # If interaction exists in available_rhs but one of its components is missing, drop interaction
    if 'Gender_Black' in available_rhs:
        if ('Gender' not in available_rhs) or ('Black' not in available_rhs):
            available_rhs = [v for v in available_rhs if v != 'Gender_Black']

    # Build formula dynamically from available RHS variables
    if not available_rhs:
        raise ValueError("No predictor variables available for modeling after excluding entirely-missing controls.")

    formula = 'Approval ~ ' + ' + '.join(available_rhs)

    # Explicit list of variables used in the model (dependent + RHS)
    model_vars = ['Approval'] + available_rhs

    # Coerce model variables to numeric and drop rows with missing values in any required variable (complete-case for selected variables)
    for v in model_vars:
        if v not in df.columns:
            raise ValueError(f"Required model variable '{v}' is missing from the dataframe.")
    model_df = df[model_vars].apply(pd.to_numeric, errors='coerce').dropna()

    if model_df.shape[0] == 0:
        raise ValueError("No observations with complete data for all selected model variables after dropping missing values. Cannot fit the model.")

    # Ensure there is still variation in the outcome after dropping incomplete rows
    if model_df['Approval'].nunique() < 2:
        raise ValueError("The dependent variable 'Approval' has no variation in the rows with complete model data. Cannot fit logistic regression.")

    # Ensure Gender varies in model_df (otherwise the Gender effect cannot be estimated)
    if model_df['Gender'].nunique() < 2:
        raise ValueError("Predictor 'Gender' has no variation in the rows with complete model data. Cannot estimate its effect.")

    # Fit logistic regression (maximum likelihood). Use only the complete-case data to avoid statsmodels encountering empty design matrices.
    logit_res = smf.logit(formula=formula, data=model_df).fit(disp=False)

    return logit_res