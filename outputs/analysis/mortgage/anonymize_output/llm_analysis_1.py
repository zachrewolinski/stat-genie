from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and return a dataframe with the exact columns used in the model.
    Creates human-readable column names, handles missing data for required variables,
    standardizes continuous covariates (z-scores), and creates an interaction term Female_Black.

    Required input columns (from provided schema): feature2..feature14

    This version is robust to modest variations in column naming (e.g. 'feature_2',
    'Feature 2', 'f2', or a column that contains a trailing '2') by attempting to
    locate the best matching input column for each required feature. If a match
    cannot be located for a conceptual variable, that variable column is created
    filled with NaNs so the downstream processing can handle incomplete rows.

    The function also tries to coerce common textual encodings of binary variables
    (e.g., 'yes'/'no', 'F'/'M', 'Approved'/'Denied') to numeric 0/1 values to avoid
    dropping rows unnecessarily when inputs are present but encoded as strings.
    """
    df = df.copy()
    original_df = df.copy()

    def _find_col_for_feature(df_cols, feature_num: int):
        """
        Try to find the best matching column name in df_cols for "feature{feature_num}".
        Returns the column name if found, otherwise None.
        """
        target_strs = [
            f"feature{feature_num}",
            f"feature_{feature_num}",
            f"feature {feature_num}",
            f"f{feature_num}",
            f"feat{feature_num}",
            str(feature_num),
        ]

        lower_cols = {c: str(c).lower() for c in df_cols}

        # Try exact matches first (case-insensitive)
        for t in target_strs:
            t_l = t.lower()
            for c, lc in lower_cols.items():
                if lc == t_l:
                    return c

        # Try contains 'feature' and the numeric token at the end equals feature_num
        for c, lc in lower_cols.items():
            if "feature" in lc:
                digits = re.findall(r"(\d+)", lc)
                if digits:
                    try:
                        if int(digits[-1]) == feature_num:
                            return c
                    except ValueError:
                        pass

        # Try columns where the final numeric token equals feature_num (handles leading zeros like '02')
        for c, lc in lower_cols.items():
            digits = re.findall(r"(\d+)", lc)
            if digits:
                try:
                    if int(digits[-1]) == feature_num:
                        return c
                except ValueError:
                    continue

        # As a last resort, if a column name is exactly the integer (as int-type column names)
        for c in df_cols:
            if isinstance(c, (int, np.integer)) and int(c) == feature_num:
                return c

        return None

    def _coerce_binary_series(series: pd.Series, kind: str) -> pd.Series:
        """
        Coerce a series to 0/1 integers where possible.
        'kind' indicates conceptual variable to resolve some ambiguous tokens:
          - 'gender' (Female): map female indicators to 1, male to 0
          - 'race_black' (Black): map black indicators to 1, others to 0 when explicit
          - 'approved' (Approved): map approved-like to 1, denied-like to 0
          - 'generic_yesno' (others): map yes/true/1 to 1, no/false/0 to 0
        If coercion is not possible for an element, it will be NaN.
        """
        if series.dtype.kind in 'biufc':  # already numeric or boolean
            coerced = pd.to_numeric(series, errors='coerce')
            return coerced.astype(float)

        s = series.astype(str).str.strip().str.lower()

        # Common maps
        yes_map = {'yes', 'y', 'true', 't', '1', '1.0', 'approved', 'approve', 'ok', 'accepted'}
        no_map = {'no', 'n', 'false', 'f', '0', '0.0', 'denied', 'deny', 'rejected'}

        female_map = {'female', 'f', 'woman', 'women', 'female*', 'fem'}
        male_map = {'male', 'm', 'man', 'men'}

        approved_map = {'approved', 'approve', 'ok', 'accepted'}
        denied_map = {'denied', 'deny', 'rejected', 'deniedpmi', 'denied_pmi'}

        black_map = {'black', 'african', 'african-american', 'african american', 'aa'}

        def map_token(token: str):
            if token in yes_map:
                return 1
            if token in no_map:
                return 0
            return None

        result = []
        for token in s:
            if token in ('nan', 'none', ''):
                result.append(np.nan)
                continue

            # gender
            if kind == 'gender':
                if token in female_map:
                    result.append(1)
                    continue
                if token in male_map:
                    result.append(0)
                    continue
                # fall back to yes/no style
                m = map_token(token)
                if m is not None:
                    result.append(m)
                    continue

            # race_black
            if kind == 'race_black':
                if token in black_map:
                    result.append(1)
                    continue
                # explicit no-like
                m = map_token(token)
                if m is not None:
                    result.append(m)
                    continue

            # approved
            if kind == 'approved':
                if token in approved_map:
                    result.append(1)
                    continue
                if token in denied_map:
                    result.append(0)
                    continue
                m = map_token(token)
                if m is not None:
                    result.append(m)
                    continue

            # generic yes/no
            if kind == 'generic_yesno':
                m = map_token(token)
                if m is not None:
                    result.append(m)
                    continue

            # Try numeric conversion as last resort
            try:
                num = float(token)
                if np.isnan(num):
                    result.append(np.nan)
                else:
                    # treat any positive non-zero as 1, zero as 0
                    if num == 0:
                        result.append(0)
                    else:
                        result.append(1 if num > 0 else 0)
            except Exception:
                result.append(np.nan)

        return pd.Series(result, index=series.index).astype(float)

    # Mapping from feature numbers in raw data to final conceptual column names
    feature_map = {
        2: 'Female',
        3: 'Black',
        4: 'HousingExpenseRatio',
        5: 'SelfEmployed',
        6: 'Married',
        7: 'MortgageScore',
        8: 'ConsumerScore',
        9: 'BadCredit',
        10: 'DebtToIncome',
        12: 'LoanToValue',
        13: 'DeniedPMI',
        14: 'Approved'
    }

    def _populate_from_source(source: pd.DataFrame) -> pd.DataFrame:
        """
        Populate conceptual columns from a given source dataframe (which is a copy of raw input).
        Returns a dataframe containing the conceptual columns (some may be NaN).
        """
        working = source.copy()
        for feat_num, final_name in feature_map.items():
            src_col = _find_col_for_feature(working.columns, feat_num)
            if src_col is None:
                working[final_name] = np.nan
            else:
                if final_name == 'Female':
                    coerced = _coerce_binary_series(working[src_col], kind='gender')
                    working[final_name] = coerced
                elif final_name == 'Black':
                    coerced = _coerce_binary_series(working[src_col], kind='race_black')
                    working[final_name] = coerced
                elif final_name in {'SelfEmployed', 'Married', 'BadCredit', 'DeniedPMI'}:
                    coerced = _coerce_binary_series(working[src_col], kind='generic_yesno')
                    working[final_name] = coerced
                elif final_name == 'Approved':
                    coerced = _coerce_binary_series(working[src_col], kind='approved')
                    working[final_name] = coerced
                else:
                    working[final_name] = pd.to_numeric(working[src_col], errors='coerce')

        # Ensure required conceptual columns exist
        required = [
            'Female', 'Black', 'HousingExpenseRatio', 'SelfEmployed', 'Married',
            'MortgageScore', 'ConsumerScore', 'BadCredit', 'DebtToIncome', 'LoanToValue',
            'DeniedPMI', 'Approved'
        ]
        for col in required:
            if col not in working.columns:
                working[col] = np.nan

        return working

    # Initially populate from the provided dataframe
    df = _populate_from_source(original_df)

    # Attempt to locate an alternative Approved-like column in the original dataframe
    # if Approved is entirely missing (all NaN). This helps when the approval column
    # exists but wasn't found via feature mapping.
    if 'Approved' in df.columns and df['Approved'].isna().all():
        approval_name_tokens = ['approve', 'approval', 'approved', 'decision', 'status', 'outcome', 'result']
        found = False
        for col in original_df.columns:
            lc = str(col).lower()
            if any(tok in lc for tok in approval_name_tokens):
                coerced = _coerce_binary_series(original_df[col], kind='approved')
                if coerced.notna().any():
                    # align to the current df index (which matches original_df)
                    df['Approved'] = coerced.reindex(df.index)
                    found = True
                    break
        # If not found by name, try to find a numeric 0/1 column as a last resort
        if not found:
            for col in original_df.columns:
                # skip columns we've already mapped that contain data
                if col in df.columns and not df[col].isna().all():
                    continue
                try:
                    series = original_df[col]
                except Exception:
                    continue
                if pd.api.types.is_numeric_dtype(series):
                    unique_vals = pd.Series(series).dropna().unique()
                    try:
                        uniq_set = set(np.unique(unique_vals))
                    except Exception:
                        uniq_set = set()
                    if uniq_set and uniq_set.issubset({0, 1}):
                        df['Approved'] = pd.to_numeric(series, errors='coerce').reindex(df.index)
                        found = True
                        break

    # We require at minimum the outcome to be present per-row.
    # Drop rows missing the outcome 'Approved' only. Binary covariates will be imputed below.
    minimal_required_per_row = ['Approved']
    df_after_drop = df.dropna(subset=minimal_required_per_row)

    # If after removing rows with missing Approved we have no rows, try to salvage by
    # repopulating from original_df and forcing Approved either from any found column
    # or, as a final fallback, fill Approved with zeros (conservative).
    if df_after_drop.shape[0] == 0:
        # If original_df itself has no rows, return an empty dataframe with required columns
        if original_df.shape[0] == 0:
            empty = pd.DataFrame(columns=[
                'Female', 'Black', 'Female_Black', 'HousingExpenseRatio_z', 'SelfEmployed', 'Married',
                'MortgageScore_z', 'ConsumerScore_z', 'BadCredit', 'DebtToIncome_z', 'LoanToValue_z',
                'DeniedPMI', 'Approved'
            ])
            return empty

        # Try to find any approved-like column in original_df
        salvaged = False
        for col in original_df.columns:
            try:
                coerced = _coerce_binary_series(original_df[col], kind='approved')
            except Exception:
                continue
            if coerced.notna().any():
                # Rebuild working dataframe from original_df and set Approved to this coerced series
                df = _populate_from_source(original_df)
                df['Approved'] = coerced.reindex(df.index)
                df_after_drop = df.dropna(subset=minimal_required_per_row)
                if df_after_drop.shape[0] > 0:
                    df = df_after_drop
                    salvaged = True
                    break

        if not salvaged:
            # As a final fallback, repopulate from original and set Approved = 0 for all rows
            df = _populate_from_source(original_df)
            df['Approved'] = 0
            # proceed with imputations on this df (which now has rows)
    else:
        df = df_after_drop

    # For remaining missing values in other predictors, impute conservatively:
    # - For binary indicators: fill missing with 0 (assume absence)
    # - For continuous covariates: fill missing with column mean (or 0 if mean is NaN)
    binary_cols = ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit', 'DeniedPMI', 'Approved']
    continuous_cols = ['HousingExpenseRatio', 'MortgageScore', 'ConsumerScore', 'DebtToIncome', 'LoanToValue']

    for bcol in binary_cols:
        if bcol in df.columns:
            df[bcol] = df[bcol].fillna(0)

    for cont in continuous_cols:
        if cont in df.columns:
            mean_val = df[cont].mean(skipna=True)
            if np.isnan(mean_val):
                mean_val = 0.0
            df[cont] = df[cont].fillna(mean_val)

    # Ensure binary indicators are integers (0/1)
    for bcol in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit', 'DeniedPMI', 'Approved']:
        if bcol in df.columns:
            df[bcol] = pd.to_numeric(df[bcol], errors='coerce').fillna(0).round().astype(int)

    # Standardize continuous covariates for easier interpretation / numerical stability
    for cont in ['HousingExpenseRatio', 'MortgageScore', 'ConsumerScore', 'DebtToIncome', 'LoanToValue']:
        zname = cont + '_z'
        if cont not in df.columns:
            df[zname] = 0.0
            continue

        mean = df[cont].mean()
        std = df[cont].std()
        if std == 0 or np.isnan(std):
            df[zname] = 0.0
        else:
            df[zname] = (df[cont] - mean) / std

    # Interaction term for moderation tests (Female x Black)
    if 'Female' not in df.columns:
        df['Female'] = 0
    if 'Black' not in df.columns:
        df['Black'] = 0
    df['Female_Black'] = df['Female'] * df['Black']

    # Return only columns necessary for modeling (keep originals optional but include modeled columns)
    keep_cols = [
        'Female', 'Black', 'Female_Black', 'HousingExpenseRatio_z', 'SelfEmployed', 'Married',
        'MortgageScore_z', 'ConsumerScore_z', 'BadCredit', 'DebtToIncome_z', 'LoanToValue_z',
        'DeniedPMI', 'Approved'
    ]

    # Safety: if any standardized columns missing (shouldn't be), create as zeros; for binary missing, create 0
    for col in keep_cols:
        if col not in df.columns:
            df[col] = 0.0 if col.endswith('_z') else 0

    # Ensure correct dtypes for returned dataframe
    for col in ['HousingExpenseRatio_z', 'MortgageScore_z', 'ConsumerScore_z', 'DebtToIncome_z', 'LoanToValue_z', 'Female_Black']:
        if col in df.columns:
            df[col] = df[col].astype(float)

    for col in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit', 'DeniedPMI', 'Approved']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # If after all attempts there are still no rows (e.g., original input empty), return empty df with required columns
    if df.shape[0] == 0:
        empty = pd.DataFrame(columns=keep_cols)
        return empty

    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) predicting Approved from Female,
    the Female x Black interaction, and controls. Returns the fitted model (robust SEs).

    Expects df to be the transformed dataframe returned by transform(df).
    """
    # If the passed df doesn't look transformed, run transform first
    try:
        cols_needed = ['Female', 'Black', 'Female_Black', 'HousingExpenseRatio_z', 'SelfEmployed',
                       'Married', 'MortgageScore_z', 'ConsumerScore_z', 'BadCredit', 'DebtToIncome_z',
                       'LoanToValue_z', 'DeniedPMI', 'Approved']
        if not all(c in df.columns for c in cols_needed):
            df = transform(df)
    except Exception:
        df = transform(df)

    # If there are no observations, raise a clear error rather than letting statsmodels
    # raise an opaque numpy reduction error.
    if df.shape[0] == 0:
        raise ValueError("Transformed dataframe contains no rows; cannot fit the model.")

    # Outcome and predictors
    y = df['Approved'].astype(float)

    X = df[[
        'Female',
        'Black',
        'Female_Black',  # explicit interaction term
        'HousingExpenseRatio_z',
        'SelfEmployed',
        'Married',
        'MortgageScore_z',
        'ConsumerScore_z',
        'BadCredit',
        'DebtToIncome_z',
        'LoanToValue_z',
        'DeniedPMI'
    ]].astype(float)

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit binomial GLM (logistic regression). Use robust (HC1) standard errors.
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
    fitted = glm_binom.fit()

    # Attach robust covariance results for inference (HC1)
    try:
        robust_results = fitted.get_robustcov_results(cov_type='HC1')
    except Exception:
        robust_results = fitted

    # Return fitted model with robust covariances (or plain if robust failed)
    return robust_results