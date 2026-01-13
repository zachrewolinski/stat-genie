from typing import Any, List, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from sklearn.linear_model import LogisticRegression


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _map_dyad_to_binary(s: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Robustly map the raw 'dyad' column to a binary ContestOutcome where:
      1 => focal group won
      0 => other group won

    Heuristics (in order):
    1. If dyad matches elementwise any other column (e.g., an ID column for the winner),
       and that column corresponds to the focal group identity (i.e., dyad equals focal-id in many rows),
       then ContestOutcome = (dyad == focal_id_column).
    2. If dyad is already numeric and in {0,1}, use as-is.
    3. If dyad is numeric with two unique values common patterns:
       - {1,2} -> map 1 -> 1, 2 -> 0
       - {-1,1} -> map 1 -> 1, -1 -> 0
       - otherwise map the smaller unique value -> 1, larger -> 0 (fallback).
    4. If dyad is string-like with two values, try to pick the value containing 'foc'/'focal' as 1.
       Otherwise map the first observed value -> 1, second -> 0 (fallback).
    5. As a last resort, treat nonzero => 1, zero => 0.
    """
    # Work on a copy
    s_clean = s.copy()

    # 1) Try to see if dyad matches any existing column elementwise (winner id)
    # If so, and matches focal-id column in a majority of rows, use that mapping.
    for col in df.columns:
        if col == s.name:
            continue
        # Compare elementwise, ignoring NA
        matches = (s_clean == df[col])
        matches_count = int(matches.sum()) if hasattr(matches, "sum") else 0
        notna_count = int(s_clean.notna().sum())
        if matches_count > 0 and notna_count > 0 and (matches_count / notna_count) > 0.5:
            # Use nullable integer so missing remain NA
            return matches.astype("Int64")

    # 2) Numeric 0/1 or numeric encodings
    if pd.api.types.is_numeric_dtype(s_clean):
        uniq = pd.Series(s_clean.dropna().unique()).sort_values().tolist()
        uniq_set = set(uniq)
        # explicit 0/1
        if uniq_set.issubset({0, 1}):
            return s_clean.fillna(pd.NA).astype("Int64")
        # common encodings
        if uniq_set == {1, 2}:
            return s_clean.map({1: 1, 2: 0}).astype("Int64")
        if uniq_set == {-1, 1}:
            return s_clean.map({1: 1, -1: 0}).astype("Int64")
        # fallback for two unique numeric values: smaller -> 1, larger -> 0
        if len(uniq) == 2:
            small, large = uniq[0], uniq[1]
            return s_clean.map({small: 1, large: 0}).astype("Int64")
        # last numeric resort: nonzero -> 1, zero -> 0
        return (s_clean != 0).map({True: 1, False: 0}).astype("Int64")

    # 4) String-like with two unique values
    if pd.api.types.is_object_dtype(s_clean) or pd.api.types.is_string_dtype(s_clean):
        uniq = pd.Series(s_clean.dropna().unique()).tolist()
        if len(uniq) == 2:
            a, b = uniq[0], uniq[1]
            a_low = str(a).lower()
            b_low = str(b).lower()
            # Prefer the value containing 'foc' or 'focal'
            if 'foc' in a_low:
                return s_clean.map({a: 1, b: 0}).astype("Int64")
            if 'foc' in b_low:
                return s_clean.map({a: 0, b: 1}).astype("Int64")
            # Prefer the value containing 'win' or 'winner'
            if 'win' in a_low or 'winner' in a_low:
                return s_clean.map({a: 1, b: 0}).astype("Int64")
            if 'win' in b_low or 'winner' in b_low:
                return s_clean.map({a: 0, b: 1}).astype("Int64")
            # Fallback deterministic mapping
            return s_clean.map({a: 1, b: 0}).astype("Int64")

    # 5) Final fallback: treat truthy as 1, falsy as 0
    return s_clean.apply(lambda x: 1 if pd.notna(x) and x not in (0, '0', False, 'False', '') else 0).astype("Int64")


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe with the required final columns.

    Required final columns (must be present in the returned dataframe):
      - ContestOutcome: 1 if focal group won, 0 if other group won
      - RelGroupSize: FocalSize - OtherSize
      - LocFocal: 1 if contest location is closer to focal group's home-center, else 0
      - FocalSize, OtherSize: absolute total group sizes
      - n_focal: number of adult males in focal group
      - other: number of adult males in other group
      - DistFocalCenter, DistOtherCenter: distances from respective home-centers to contest location
    """
    df = df.copy()

    # ---------- ContestOutcome ----------
    if 'dyad' not in df.columns:
        raise ValueError("Expected column 'dyad' in input dataframe")

    df['ContestOutcome'] = _map_dyad_to_binary(df['dyad'], df)

    # ---------- Distances ----------
    # Try multiple candidate source column names for robustness
    dist_focal_candidates = ['win', 'dist_focal', 'dist_focal_center', 'DistFocalCenter',
                             'distfocal', 'dist_focalcenter', 'dist_to_focal', 'dist_focal_home']
    dist_other_candidates = ['m_focal', 'dist_other', 'dist_other_center', 'DistOtherCenter',
                             'distother', 'dist_othercenter', 'dist_to_other', 'dist_other_home']

    src_dist_focal = _first_existing_column(df, dist_focal_candidates)
    src_dist_other = _first_existing_column(df, dist_other_candidates)

    if src_dist_focal is not None:
        df['DistFocalCenter'] = pd.to_numeric(df[src_dist_focal], errors='coerce')
    else:
        # If not found, create column filled with NaN so downstream missing-value handling will remove rows
        df['DistFocalCenter'] = np.nan

    if src_dist_other is not None:
        df['DistOtherCenter'] = pd.to_numeric(df[src_dist_other], errors='coerce')
    else:
        df['DistOtherCenter'] = np.nan

    # ---------- Group sizes ----------
    focal_size_candidates = ['f_focal', 'focal_size', 'FocalSize', 'size_focal', 'size_foc', 'focal_total', 'f_total']
    other_size_candidates = ['f_other', 'other_size', 'OtherSize', 'size_other', 'other_total', 'f_other_total']

    src_focal_size = _first_existing_column(df, focal_size_candidates)
    src_other_size = _first_existing_column(df, other_size_candidates)

    if src_focal_size is not None:
        df['FocalSize'] = pd.to_numeric(df[src_focal_size], errors='coerce')
    else:
        df['FocalSize'] = np.nan

    if src_other_size is not None:
        df['OtherSize'] = pd.to_numeric(df[src_other_size], errors='coerce')
    else:
        df['OtherSize'] = np.nan

    # Relative size
    df['RelGroupSize'] = df['FocalSize'] - df['OtherSize']

    # Relative size ratio for diagnostics (not required by model)
    df['RelSizeRatio'] = df['FocalSize'] / df['OtherSize'].replace({0: np.nan})

    # ---------- Location advantage ----------
    # If distances are available, compute LocFocal as 1 if DistFocalCenter < DistOtherCenter
    # If distances are missing, try to derive from any existing 'loc_focal' or similar column
    if df['DistFocalCenter'].notna().any() or df['DistOtherCenter'].notna().any():
        # If both distances are present, compare; where one is missing result will be False -> set NA
        loc_series = pd.Series(np.nan, index=df.index, dtype="float64")
        mask_both = df['DistFocalCenter'].notna() & df['DistOtherCenter'].notna()
        loc_series[mask_both] = (df.loc[mask_both, 'DistFocalCenter'] < df.loc[mask_both, 'DistOtherCenter']).astype(int)
        # Cast to nullable integer
        df['LocFocal'] = loc_series.astype("Int64")
    else:
        # Try alternative candidate columns for location advantage
        loc_candidates = ['loc_focal', 'LocFocal', 'at_home', 'home_advantage', 'focal_at_home']
        src_loc = _first_existing_column(df, loc_candidates)
        if src_loc is not None:
            df['LocFocal'] = pd.to_numeric(df[src_loc], errors='coerce').fillna(0).astype("Int64")
        else:
            # If no information, default to NaN so rows will be dropped
            df['LocFocal'] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # ---------- Male counts (controls) ----------
    n_focal_candidates = ['n_focal', 'n_focal_males', 'n_males_focal', 'n_foc', 'n_male_focal', 'males_focal']
    other_male_candidates = ['other', 'n_other', 'n_males_other', 'n_male_other', 'males_other', 'n_other_males']

    src_n_focal = _first_existing_column(df, n_focal_candidates)
    src_other_males = _first_existing_column(df, other_male_candidates)

    if src_n_focal is not None:
        df['n_focal'] = pd.to_numeric(df[src_n_focal], errors='coerce')
    else:
        df['n_focal'] = np.nan

    if src_other_males is not None:
        # If the source column is actually named 'other' and already exists, this will overwrite it with numeric values.
        df['other'] = pd.to_numeric(df[src_other_males], errors='coerce')
    else:
        df['other'] = np.nan

    # Optional derived control
    df['MaleDiff'] = df['n_focal'] - df['other']

    # ---------- Final required columns check and drop NA ----------
    required_cols = [
        'ContestOutcome', 'RelGroupSize', 'LocFocal',
        'n_focal', 'other', 'FocalSize', 'OtherSize',
        'DistFocalCenter', 'DistOtherCenter'
    ]

    df_final = df.dropna(subset=required_cols).copy()

    # Ensure ContestOutcome is integer 0/1
    # Convert nullable Int64 to regular int dtype after dropping NA
    if not set(pd.Series(df_final['ContestOutcome'].unique()).tolist()).issubset({0, 1, pd.NA}):
        # If any values outside {0,1} remain, attempt a final coercion (nonzero -> 1)
        df_final['ContestOutcome'] = df_final['ContestOutcome'].apply(lambda x: 1 if x not in (0, '0', False, None) else 0).astype(int)
    else:
        # Safe cast: ContestOutcome may be Int64 (nullable) -> convert to non-nullable int
        df_final['ContestOutcome'] = df_final['ContestOutcome'].astype(int)

    # Ensure numeric columns are numeric types
    numeric_cols = ['RelGroupSize', 'FocalSize', 'OtherSize', 'n_focal', 'other', 'DistFocalCenter', 'DistOtherCenter']
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    # LocFocal should be integer 0/1
    df_final['LocFocal'] = df_final['LocFocal'].astype(int)

    return df_final


def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial) predicting the probability the focal group wins.

    Main specification:
      ContestOutcome ~ RelGroupSize * LocFocal + n_focal + other

    Returns the fitted statsmodels results object when possible; if that fails
    due to numerical issues (perfect separation / singular matrix), falls back
    to a regularized scikit-learn LogisticRegression estimator.

    If the outcome contains only a single class (e.g., all 1s or all 0s),
    returns a simple constant predictor object rather than attempting to fit.
    """
    # Ensure the dataframe contains needed columns
    for c in ['ContestOutcome', 'RelGroupSize', 'LocFocal', 'n_focal', 'other']:
        if c not in df.columns:
            raise ValueError(f"Column {c} required for modeling but not found in dataframe")

    # Ensure ContestOutcome is 0/1
    unique_contest = pd.Series(df['ContestOutcome'].dropna().unique()).tolist()
    if not set(unique_contest).issubset({0, 1}):
        raise ValueError("ContestOutcome must be binary (0/1) for logistic regression")

    # If only one class present, return a constant predictor object
    unique_nonmissing = sorted(set(pd.Series(df['ContestOutcome'].dropna().unique()).tolist()))
    if len(unique_nonmissing) == 0:
        raise ValueError("No non-missing ContestOutcome values available for modeling")
    if len(unique_nonmissing) == 1:
        single_class = int(unique_nonmissing[0])

        class ConstantModel:
            def __init__(self, const):
                self.constant = int(const)
                # Provide sklearn-like attributes for downstream inspection
                self.coef_ = np.zeros((1, 0))
                # Use large finite intercept to reflect near-certain prediction
                self.intercept_ = np.array([1000.0]) if self.constant == 1 else np.array([-1000.0])
                self.classes_ = np.array([self.constant])

            def predict(self, X):
                # X can be array-like or dataframe; return constant for each row
                length = 0
                try:
                    length = len(X)
                except Exception:
                    length = 1
                return np.full(length, self.constant, dtype=int)

            def predict_proba(self, X):
                length = 0
                try:
                    length = len(X)
                except Exception:
                    length = 1
                if self.constant == 1:
                    return np.vstack([np.zeros(length), np.ones(length)]).T
                else:
                    return np.vstack([np.ones(length), np.zeros(length)]).T

            def __repr__(self):
                return f"<ConstantModel predict={self.constant}>"

        const_model = ConstantModel(single_class)
        print(f"Outcome contains only a single class ({single_class}). Returning constant predictor.")
        return const_model

    formula = 'ContestOutcome ~ RelGroupSize * LocFocal + n_focal + other'
    logit_model = smf.logit(formula=formula, data=df)

    try:
        # Try the canonical maximum likelihood fit first
        results = logit_model.fit(disp=False)
    except (np.linalg.LinAlgError, PerfectSeparationError, ValueError) as e:
        # If statsmodels fitting fails due to singularity / perfect separation / value errors,
        # fall back to a numerically stable regularized fit using scikit-learn.
        # Build the design matrix consistent with the formula:
        # include RelGroupSize, LocFocal, their interaction, n_focal, other
        y = pd.to_numeric(df['ContestOutcome'], errors='coerce').astype(float).values
        X_df = pd.DataFrame({
            'RelGroupSize': pd.to_numeric(df['RelGroupSize'], errors='coerce'),
            'LocFocal': pd.to_numeric(df['LocFocal'], errors='coerce'),
            'n_focal': pd.to_numeric(df['n_focal'], errors='coerce'),
            'other': pd.to_numeric(df['other'], errors='coerce')
        }, index=df.index)
        X_df['RelGroupSize:LocFocal'] = X_df['RelGroupSize'] * X_df['LocFocal']

        # Drop any rows with NaN in X or y (should be none if transform worked correctly)
        mask = (~X_df.isna().any(axis=1)) & (~pd.isna(y))
        X_clean = X_df.loc[mask]
        y_clean = y[mask]

        if X_clean.shape[0] == 0:
            raise ValueError("No observations available for modeling after preparing design matrix")

        # Check that y_clean contains at least two classes
        classes_in_y = np.unique(y_clean)
        if len(classes_in_y) < 2:
            single_class = int(classes_in_y[0])
            class ConstantModel:
                def __init__(self, const):
                    self.constant = int(const)
                    self.coef_ = np.zeros((1, X_clean.shape[1]))
                    self.intercept_ = np.array([1000.0]) if self.constant == 1 else np.array([-1000.0])
                    self.classes_ = np.array([self.constant])

                def predict(self, X):
                    length = 0
                    try:
                        length = len(X)
                    except Exception:
                        length = 1
                    return np.full(length, self.constant, dtype=int)

                def predict_proba(self, X):
                    length = 0
                    try:
                        length = len(X)
                    except Exception:
                        length = 1
                    if self.constant == 1:
                        return np.vstack([np.zeros(length), np.ones(length)]).T
                    else:
                        return np.vstack([np.ones(length), np.zeros(length)]).T

                def __repr__(self):
                    return f"<ConstantModel predict={self.constant}>"

            const_model = ConstantModel(single_class)
            print(f"After preparing design matrix, outcome contains only a single class ({single_class}). Returning constant predictor.")
            return const_model

        # Fit a regularized logistic regression (L2) via scikit-learn for numerical stability.
        # Use a small amount of regularization (C inverse regularization strength) to avoid singularities.
        clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
        clf.fit(X_clean.values, y_clean)

        # Return the sklearn estimator as the fallback result object
        results = clf

    # Print summary when available, but do not error if the returned object lacks summary()
    try:
        print(results.summary())
    except Exception:
        # If sklearn estimator, print coefficients for quick inspection
        try:
            coef = getattr(results, "coef_", None)
            intercept = getattr(results, "intercept_", None)
            if coef is not None:
                print("Fallback sklearn LogisticRegression fitted. Coefficients:", coef, "Intercept:", intercept)
            else:
                print(results)
        except Exception:
            print(results)

    return results