from typing import Any
import numpy as np
import pandas as pd


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Coerce commonly used numeric columns to numeric where appropriate
    for col in ['f_focal', 'f_other', 'm_focal', 'n_focal', 'other', 'dyad', 'm_other']:
        if col in df.columns:
            df[col] = pd.to_numeric(df.get(col), errors='coerce')

    # -----------------------
    # Dependent variable: Win
    # -----------------------
    # Heuristic to locate/construct the Win column (must be 0/1 in final dataframe)
    win_col_candidates = ['Win', 'win', 'winner', 'outcome', 'result', 'focal_win']
    win_series = None

    for c in win_col_candidates:
        if c in df.columns:
            s = df[c]
            # If values appear to be numeric 0/1 or boolean-like, coerce
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                # Convert numeric/bool to 0/1 integers where possible
                s_num = pd.to_numeric(s, errors='coerce')
                unique_vals = pd.Series(s_num).dropna().unique()
                # If the unique numeric values are subset of {0,1} (or empty), accept
                if set(np.unique(unique_vals)).issubset({0, 1}):
                    win_series = s_num.fillna(pd.NA).astype('Int64')
                    break
            # If string values like 'focal'/'other' or 'Focal' etc., map them
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
                mapped = s.map({
                    'focal': 1, 'Focal': 1, 'F': 1, 'f': 1, 'focal_win': 1,
                    'other': 0, 'Other': 0, 'O': 0, 'o': 0, 'loss': 0,
                    'win': 1, 'Win': 1, 'winner': 1, 'winner_focal': 1
                })
                if mapped.notna().any():
                    win_series = pd.to_numeric(mapped, errors='coerce').astype('Int64')
                    break

    # As a last resort, if no suitable Win column found but there is a 'dyad' that looks binary, use it
    if win_series is None and 'dyad' in df.columns:
        dy = pd.to_numeric(df['dyad'], errors='coerce')
        unique_vals = pd.Series(dy).dropna().unique()
        if set(np.unique(unique_vals)).issubset({0, 1}):
            win_series = dy.astype('Int64')

    # If still None, create a Win column filled with NA (will be handled later)
    if win_series is None:
        df['Win'] = pd.Series([pd.NA] * len(df), index=df.index, dtype='Int64')
    else:
        df['Win'] = win_series

    # -----------------------
    # m_other: clustering id
    # -----------------------
    # Ensure final dataframe includes 'm_other' (dyad/pair id used for clustering)
    if 'm_other' in df.columns:
        df['m_other'] = pd.to_numeric(df['m_other'], errors='coerce')
    elif 'dyad' in df.columns:
        df['m_other'] = pd.to_numeric(df['dyad'], errors='coerce')
    else:
        # If neither present, create missing column of NA (will be handled later)
        df['m_other'] = pd.Series([pd.NA] * len(df), index=df.index)

    # -----------------------
    # Group size variables
    # -----------------------
    # focal_total and other_total come from f_focal and f_other in the raw data
    df['focal_total'] = pd.to_numeric(df.get('f_focal'), errors='coerce')
    df['other_total'] = pd.to_numeric(df.get('f_other'), errors='coerce')

    # Controls: number of males in each group (rename for clarity)
    df['n_males_focal'] = pd.to_numeric(df.get('n_focal'), errors='coerce')
    df['n_males_other'] = pd.to_numeric(df.get('other'), errors='coerce')

    # -----------------------
    # Relative size measures
    # -----------------------
    # Ratio: focal / other
    # Avoid divide-by-zero by coercing zeros to NaN for denominator
    denom = df['other_total'].replace({0: np.nan})
    df['relative_size_ratio'] = df['focal_total'] / denom
    # Replace infinities (division by zero) with NaN so they will be handled
    df['relative_size_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # -----------------------
    # Distances and adv_home
    # -----------------------
    # Try to find the best candidate columns for focal_dist and other_dist
    def first_existing_numeric(cols):
        for c in cols:
            if c in df.columns:
                ser = pd.to_numeric(df[c], errors='coerce')
                if ser.notna().any():
                    return ser
        return None

    focal_dist_candidates = ['focal_dist', 'focal_distance', 'distance_focal']
    other_dist_candidates = ['other_dist', 'other_distance', 'distance_other', 'm_focal']

    focal_dist_ser = first_existing_numeric(focal_dist_candidates)
    other_dist_ser = first_existing_numeric(other_dist_candidates)

    # If both found, compute adv_home = other_dist - focal_dist
    if focal_dist_ser is not None and other_dist_ser is not None:
        df['focal_dist'] = focal_dist_ser
        df['other_dist'] = other_dist_ser
        df['adv_home'] = df['other_dist'] - df['focal_dist']
        df['adv_home'] = pd.to_numeric(df['adv_home'], errors='coerce')
    else:
        # Create adv_home as NA if distances not available
        df['focal_dist'] = pd.Series([pd.NA] * len(df), index=df.index)
        df['other_dist'] = pd.Series([pd.NA] * len(df), index=df.index)
        df['adv_home'] = pd.Series([pd.NA] * len(df), index=df.index)

    # -----------------------
    # Categorical location dummies
    # -----------------------
    # Positive adv_home => focal nearer its center than other (advantage)
    # Define neutral band +/-10 units
    # Ensure adv_home numeric
    df['adv_home'] = pd.to_numeric(df['adv_home'], errors='coerce')
    df['location'] = np.where(df['adv_home'] > 10, 'FocalHome',
                              np.where(df['adv_home'] < -10, 'OtherHome', 'Neutral'))
    loc_dummies = pd.get_dummies(df['location'], prefix='Location')
    # Drop Neutral to serve as reference if present
    if 'Location_Neutral' in loc_dummies.columns:
        loc_dummies = loc_dummies.drop(columns=['Location_Neutral'])
    df = pd.concat([df, loc_dummies], axis=1)

    # Ensure the required location dummy columns exist in final dataframe (create with zeros if absent)
    for col in ['Location_FocalHome', 'Location_OtherHome']:
        if col not in df.columns:
            df[col] = 0

    # -----------------------
    # Interaction term (internal helper; not required by model)
    # -----------------------
    df['size_adv_interaction'] = df['relative_size_ratio'] * df['adv_home']

    # -----------------------
    # Final required columns and cleanup
    # -----------------------
    # The essential conceptual variables that must be present for modeling:
    # Note: adv_home is a required final column but may be missing for some rows;
    # to avoid dropping all rows when adv_home is unavailable we do not force
    # completeness of adv_home here. We require outcome and core size predictors.
    dropna_subset = ['Win', 'relative_size_ratio', 'focal_total', 'n_males_focal', 'n_males_other']
    present_subset = [c for c in dropna_subset if c in df.columns]
    # Drop rows with missing values in these essential predictors/outcome;
    # allow optional predictors like 'adv_home' and 'size_adv_interaction' to be missing.
    df = df.dropna(subset=present_subset)

    # Ensure Win is exactly 0/1 integers in the final dataframe
    # If Win contains other values, coerce to binary: treat value==1 as 1, everything else as 0
    # (After dropping rows with missing Win above, this conversion is safe.)
    def _win_to_binary(x):
        try:
            if pd.isna(x):
                return 0
            xi = int(x)
            return 1 if xi == 1 else 0
        except Exception:
            # If cannot convert, treat as 0
            return 0

    df['Win'] = df['Win'].apply(lambda x: 1 if pd.notna(x) and int(x) == 1 else 0).astype('int64')

    # Reset index and return
    df = df.reset_index(drop=True)
    return df


def model(df: pd.DataFrame) -> Any:
    import statsmodels.api as sm

    # Ensure required dummy columns exist (if a category was absent in this sample, create zero column)
    for col in ['Location_FocalHome', 'Location_OtherHome']:
        if col not in df.columns:
            df[col] = 0

    # Predictor columns: main effects, location dummies, and controls
    # Note: Do NOT include internal helper 'size_adv_interaction' as a required model predictor.
    X_cols = [
        'relative_size_ratio',
        'adv_home',
        'Location_FocalHome',
        'Location_OtherHome',
        'focal_total',
        'n_males_focal',
        'n_males_other'
    ]

    # Verify predictors exist in df
    X_cols = [c for c in X_cols if c in df.columns]

    if len(X_cols) == 0:
        raise RuntimeError("No predictor columns available in dataframe to build design matrix.")

    # Build design matrix and dependent variable, ensuring float types
    X = df[X_cols].astype(float)

    # Drop predictors that are entirely missing (all-NaN) so they don't force row drops
    X = X.dropna(axis=1, how='all')

    if X.shape[1] == 0:
        raise RuntimeError("No predictor columns with any finite values are available to build the design matrix.")

    # Add constant
    X_const = sm.add_constant(X, has_constant='add').astype(float)

    # Ensure dependent variable is 0/1 floats
    if 'Win' not in df.columns:
        raise RuntimeError("Dependent variable 'Win' not found in dataframe.")
    y = (df['Win'] == 1).astype(float)

    # Align and replace infinities with NaN
    data = pd.concat([X, y.rename('Win'), df['m_other']], axis=1)
    data = data.replace([np.inf, -np.inf], np.nan)

    # Determine which predictors actually have any finite values in the assembled data
    required_predictors = [c for c in X.columns if data[c].notna().any()]

    if len(required_predictors) == 0:
        raise ValueError("No predictors have any finite values in the provided dataframe. "
                         "Ensure at least one predictor contains finite data.")

    # Drop rows with missing values in the predictors that have any finite values, or outcome
    data = data.dropna(axis=0, how='any', subset=required_predictors + ['Win'])

    # If no rows remain, attempt to salvage by selecting rows where outcome is present and at least one predictor is finite
    if data.shape[0] == 0:
        # Rows where outcome is finite
        mask_y = y.notna()
        # Identify rows where each predictor is finite and outcome is finite
        valid_rows_union = pd.Series(False, index=df.index)
        predictor_has_valid = []
        for c in X.columns:
            col_vals = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            mask = col_vals.notna() & mask_y
            if mask.any():
                predictor_has_valid.append(c)
                valid_rows_union = valid_rows_union | mask

        if not valid_rows_union.any() or len(predictor_has_valid) == 0:
            # No usable rows found: raise informative error
            raise ValueError("No data available after dropping rows with non-finite values in predictors or outcome. "
                             "Ensure the input dataframe contains finite values for at least one predictor and the outcome 'Win'.")

        # Restrict to rows with outcome and at least one valid predictor
        valid_idx = valid_rows_union[valid_rows_union].index

        # Select predictors that have any valid values within these rows
        selected_predictors = []
        for c in predictor_has_valid:
            col_vals = pd.to_numeric(df.loc[valid_idx, c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            if col_vals.notna().any():
                selected_predictors.append(c)

        if len(selected_predictors) == 0:
            raise ValueError("No data available after dropping rows with non-finite values in predictors or outcome (after reducing predictors).")

        # Rebuild reduced design matrix with constant
        X_reduced = df.loc[valid_idx, selected_predictors].astype(float)
        X_reduced = X_reduced.dropna(axis=1, how='all')
        if X_reduced.shape[1] == 0:
            raise ValueError("No predictor columns with finite values are available after reduction.")

        X_reduced_const = sm.add_constant(X_reduced, has_constant='add').astype(float)

        # Build data with reduced predictors and available rows
        data = pd.concat([X_reduced_const, y.loc[valid_idx].rename('Win'), df.loc[valid_idx, 'm_other']], axis=1)
        data = data.replace([np.inf, -np.inf], np.nan)
        # Drop any rows that still have missing values in the reduced predictors or outcome
        data = data.dropna(axis=0, how='any', subset=list(X_reduced_const.columns) + ['Win'])
        if data.shape[0] == 0:
            raise ValueError("No data available after dropping rows with non-finite values in predictors or outcome (after reducing predictors).")

        # Update X and X_const for the reduced set
        X = X_reduced
        X_const = X_reduced_const

    # Ensure X_fit columns exist in data; add constant if necessary
    # If X_const contains 'const' but data lacks it (shouldn't after salvage), create it
    if 'const' in X_const.columns and 'const' not in data.columns:
        data['const'] = 1.0

    # Use the columns from X_const that are present in data for fitting
    fit_cols = [c for c in X_const.columns if c in data.columns]
    if len(fit_cols) == 0:
        raise RuntimeError("No independent predictors available to fit the model after processing.")

    X_fit = data[fit_cols].astype(float)
    y_fit = data['Win']
    groups_fit = data['m_other'] if 'm_other' in data.columns else None

    # Check that the dependent variable is not constant
    if y_fit.nunique() < 2:
        raise ValueError("Dependent variable 'Win' is constant in the sample; cannot fit logistic regression.")

    # If columns are collinear (singular matrix), remove collinear columns using QR pivoting
    def select_independent_cols(X_df: pd.DataFrame):
        X_mat = X_df.values
        ncols = X_mat.shape[1]
        rank = np.linalg.matrix_rank(X_mat)
        if rank == ncols:
            return list(X_df.columns)
        # Try QR pivoting to pick independent columns
        try:
            Q, R, piv = np.linalg.qr(X_mat, mode='reduced', pivoting=True)
            keep = list(X_df.columns[piv[:rank]])
            return keep
        except Exception:
            # Fallback greedy approach: iteratively remove columns that do not reduce rank
            cols = list(X_df.columns)
            while True:
                Xm = X_df[cols].values
                r = np.linalg.matrix_rank(Xm)
                if r == len(cols) or len(cols) == 0:
                    break
                removed = False
                for c in cols:
                    cols_try = [cc for cc in cols if cc != c]
                    if len(cols_try) == 0:
                        continue
                    if np.linalg.matrix_rank(X_df[cols_try].values) == r:
                        cols.remove(c)
                        removed = True
                        break
                if not removed:
                    break
            final_rank = np.linalg.matrix_rank(X_df[cols].values)
            return cols[:final_rank]

    keep_cols = select_independent_cols(X_fit)
    if len(keep_cols) == 0:
        raise RuntimeError("No independent predictors available to fit the model after removing collinear columns.")

    X_fit = X_fit[keep_cols]

    # Fit a logistic regression (binomial) predicting the probability focal group wins
    logit = sm.Logit(y_fit, X_fit)

    # Attempt standard fit; if it fails due to singularity or separation, fall back to a regularized fit
    try:
        res = logit.fit(disp=False)
    except Exception:
        # Try a regularized fit (small penalty) to handle separation or singularities
        try:
            res = logit.fit_regularized(method='l1', alpha=1e-6, maxiter=1000)
        except Exception:
            try:
                res = logit.fit_regularized(method='l2', alpha=1e-6, maxiter=1000)
            except Exception as e:
                # As a last resort, raise the original error
                raise RuntimeError(f"Model fitting failed: {e}")

    # Obtain cluster-robust standard errors clustered by dyad/pair id (m_other) if possible
    res_robust = res
    try:
        if groups_fit is not None and groups_fit.notna().all() and groups_fit.nunique() > 1:
            if hasattr(res, 'get_robustcov_results'):
                res_robust = res.get_robustcov_results(cov_type='cluster', groups=groups_fit)
    except Exception:
        # Fall back to default results if clustering fails
        res_robust = res

    # Return the fitted model with robust covariances where available
    return res_robust