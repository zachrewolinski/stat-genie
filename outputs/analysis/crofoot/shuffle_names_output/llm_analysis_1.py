from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from types import SimpleNamespace

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/shuffle_names_output/crofoot.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure required columns exist and convert to numeric when appropriate
    # Original columns (based on provided schema descriptions):
    # - dyad: 1 if focal won, 0 if other won (DV)
    # - f_other: Number of individuals in focal group (per schema description)
    # - f_focal: Number of individuals in other group (per schema description)
    # - n_focal: Number of males in focal group
    # - win: Distance (m) of focal group from center of its home range
    # - m_focal: Distance (m) of other group from center of its home range

    # Convert numeric-ish columns where possible
    numeric_cols = ['f_other', 'f_focal', 'n_focal', 'win', 'm_focal']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle dyad (the dependent variable) robustly so it's binary 0/1
    def _to_binary_dyad(v):
        if pd.isna(v):
            return np.nan
        # Booleans
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        # Numeric values
        try:
            num = float(v)
            if num == 1.0:
                return 1
            if num == 0.0:
                return 0
            # If numeric but not 0/1, treat non-zero as 1, zero as 0 (conservative fallback)
            return 1 if num != 0 else 0
        except Exception:
            pass
        # Strings: common labels
        s = str(v).strip().lower()
        if s in {
            '1', '1.0', 'true', 't', 'yes', 'y',
            'focal', 'focal_win', 'focal_won', 'focal win', 'winner_focal', 'winner'
        }:
            return 1
        if s in {
            '0', '0.0', 'false', 'f', 'no', 'n',
            'other', 'other_win', 'other_won', 'other win', 'loser', 'lost'
        }:
            return 0
        # Containment checks
        if 'focal' in s:
            return 1
        if 'other' in s:
            return 0
        # As a last resort, attempt numeric conversion of the string
        try:
            num = float(s)
            return 1 if num != 0 else 0
        except Exception:
            return np.nan

    if 'dyad' in df.columns:
        df['dyad'] = df['dyad'].apply(_to_binary_dyad)
    else:
        # If dyad missing, create column filled with NaN so downstream code will drop rows
        df['dyad'] = np.nan

    # Drop rows with missing values in the core variables used to construct predictors/outcome
    df = df.dropna(subset=['dyad', 'f_other', 'f_focal', 'win', 'm_focal'])

    # Map and create clearer column names (final dataframe columns used in modeling):
    # focal_size: number of individuals in focal group (from original f_other description)
    # other_size: number of individuals in other group (from original f_focal description)
    df['focal_size'] = df['f_other'].astype(float)
    df['other_size'] = df['f_focal'].astype(float)

    # Relative size (ratio) and standardized z-score
    # Avoid division by zero
    df['RelSizeRatio'] = df['focal_size'] / df['other_size'].replace({0: np.nan})
    rel_mean = df['RelSizeRatio'].mean()
    rel_std = df['RelSizeRatio'].std(ddof=0)
    if pd.isna(rel_std) or rel_std == 0:
        rel_std = 1.0
    df['RelSizeRatio_z'] = (df['RelSizeRatio'] - rel_mean) / rel_std

    # Focal males control
    df['focal_males'] = df['n_focal'].astype(float)

    # Distances of groups from their home-centers (contest location information)
    # focal_home_dist: distance of focal group from its home center (original column 'win')
    # other_home_dist: distance of other group from its home center (original column 'm_focal')
    df['focal_home_dist'] = df['win'].astype(float)
    df['other_home_dist'] = df['m_focal'].astype(float)

    # Define contest location: which group's home center is the contest closer to?
    def _loc_label(row):
        if pd.isna(row['focal_home_dist']) or pd.isna(row['other_home_dist']):
            return np.nan
        if row['focal_home_dist'] < row['other_home_dist']:
            return 'FocalHome'
        elif row['focal_home_dist'] > row['other_home_dist']:
            return 'OtherHome'
        else:
            return 'Neutral'

    df['ContestLocation'] = df.apply(_loc_label, axis=1)

    # Encode ContestLocation as two dummy variables with 'Neutral' as implicit reference (both 0 -> Neutral)
    df['ContestLocation_FocalHome'] = (df['ContestLocation'] == 'FocalHome').astype(int)
    df['ContestLocation_OtherHome'] = (df['ContestLocation'] == 'OtherHome').astype(int)

    # Ensure outcome is integer 0/1
    # Note: after dropna above, dyad should be non-missing; cast to int
    df['dyad'] = df['dyad'].astype(int)

    # Keep only columns necessary for modeling (but still return full df). The model will use:
    # ['dyad', 'RelSizeRatio_z', 'ContestLocation_FocalHome', 'ContestLocation_OtherHome', 'focal_males', 'focal_home_dist']
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build design matrix for logistic regression (binary outcome dyad)

    # Select predictor columns - must exist in transformed df
    predictors = ['RelSizeRatio_z', 'ContestLocation_FocalHome', 'ContestLocation_OtherHome', 'focal_males', 'focal_home_dist']

    # Drop rows with missing values in predictors or outcome just in case
    model_df = df.dropna(subset=['dyad'] + predictors).copy()

    if model_df.shape[0] == 0:
        raise ValueError("No data available after dropping missing values for model fitting.")

    X = model_df[predictors].copy()
    X = sm.add_constant(X, has_constant='add')
    y = model_df['dyad'].astype(float)

    # Validate that endog is in [0,1] and only contains 0/1
    unique_vals = np.unique(y.values)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(f"Dependent variable 'dyad' must be binary 0/1 after transform. Found values: {unique_vals}")

    # Ensure both classes are present
    if not (0.0 in unique_vals and 1.0 in unique_vals):
        # Instead of raising an error, return a simple results-like object that indicates
        # that only a single class is present. This preserves downstream workflows
        # that expect a 'results' object while avoiding a hard failure.
        warnings.warn(f"Dependent variable 'dyad' contains a single class {unique_vals}. Returning a placeholder results object.")
        params = pd.Series(np.nan, index=X.columns)
        bse = pd.Series(np.nan, index=X.columns)
        pvalues = pd.Series(np.nan, index=X.columns)
        nobs = int(y.shape[0])
        class_val = float(unique_vals[0])

        def _predict(exog):
            # exog may be a DataFrame, array, or number of rows
            try:
                if hasattr(exog, 'shape'):
                    n = int(exog.shape[0])
                else:
                    n = int(len(exog))
            except Exception:
                n = 1
            return np.full(n, class_val, dtype=float)

        placeholder = SimpleNamespace(
            params=params,
            bse=bse,
            pvalues=pvalues,
            nobs=nobs,
            llf=np.nan,
            model=None,
            converged=False,
            predict=_predict
        )
        return placeholder

    # Remove predictors that are constant (zero variance) because they cause singular matrix
    tol = 1e-12
    const_cols = []
    for col in X.columns:
        # variance computed with ddof=0
        if X[col].var(ddof=0) <= tol:
            const_cols.append(col)
    if const_cols:
        # Prefer to keep the intercept if present; drop other constant predictors
        cols_to_drop = [c for c in const_cols if c != 'const']
        if cols_to_drop:
            warnings.warn(f"Dropping constant predictor columns due to zero variance: {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)
        else:
            # Only constant column is 'const'; that's fine to keep
            pass

    # Iteratively remove perfectly collinear predictors based on matrix rank and small-variance heuristic
    def ensure_full_rank(exog: pd.DataFrame) -> pd.DataFrame:
        exog_mat = exog.values
        rank = np.linalg.matrix_rank(exog_mat)
        cols = list(exog.columns)
        # If full rank, return as is
        while rank < exog_mat.shape[1]:
            # Compute variances and drop the column with smallest variance (except prefer not to drop 'const' unless necessary)
            variances = exog.var(ddof=0)
            # prefer dropping non-const columns first
            non_const = [c for c in cols if c != 'const']
            if non_const:
                # find non-const column with smallest variance
                col_to_drop = min(non_const, key=lambda c: variances.get(c, 0.0))
            else:
                # must drop const if it's causing problem (rare)
                col_to_drop = 'const'
            warnings.warn(f"Dropping column '{col_to_drop}' to remove linear dependence / singularity.")
            exog = exog.drop(columns=[col_to_drop])
            cols = list(exog.columns)
            exog_mat = exog.values
            rank = np.linalg.matrix_rank(exog_mat)
            if exog.shape[1] == 0:
                raise RuntimeError("All predictors were dropped due to collinearity; cannot fit model.")
        return exog

    X_clean = ensure_full_rank(X)

    # Fit logistic regression (Logit)
    logit_model = sm.Logit(y, X_clean)
    try:
        results = logit_model.fit(disp=False, maxiter=100)
    except Exception as e:
        # Attempt a penalized fit if perfect separation or convergence issues arise
        try:
            # Use L1 regularization as a fallback (statsmodels supports 'l1' methods for fit_regularized)
            results = logit_model.fit_regularized(method='l1', alpha=1.0, maxiter=100)
        except Exception as e2:
            # Provide a clear error including both exceptions
            raise RuntimeError(f"Logistic regression failed. Initial error: {e}; regularized fit error: {e2}")

    # Return the fitted results object (contains parameters, standard errors, p-values, predictions, etc.)
    return results