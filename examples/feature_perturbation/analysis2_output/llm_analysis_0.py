import pandas as pd
import numpy as np
from typing import Any, List


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for modeling.

    Behavior:
    - Ensures an 'outcome' column exists. If one of the common outcome column names
      ('outcome', 'y', 'target', 'label', 'response', 'dependent') exists, a copy
      of the first found is placed in df['outcome'] (unless 'outcome' already exists).
    - If no obvious outcome column exists, picks the first numeric column as outcome.
    - If there are no numeric columns, factorizes the first column and uses that as outcome.
    - Returns a new dataframe (does not modify the input in-place).
    """
    df = df.copy()

    # Candidate names to search for (in order)
    outcome_candidates: List[str] = ["outcome", "y", "target", "label", "response", "dependent"]

    # Find any existing candidate column (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    found_name = None
    for cand in outcome_candidates:
        if cand.lower() in col_map:
            found_name = col_map[cand.lower()]
            break

    if found_name:
        # If 'outcome' does not exist, create it as a copy of the found column.
        if "outcome" not in df.columns:
            df["outcome"] = df[found_name].copy()
        else:
            # If 'outcome' exists, leave it as-is (do not overwrite).
            # But ensure there's at least one numeric outcome: if outcome is non-numeric, try to coerce.
            if not pd.api.types.is_numeric_dtype(df["outcome"]):
                coerced = pd.to_numeric(df["outcome"], errors="coerce")
                if coerced.isna().all():
                    # fallback: factorize original found column
                    df["outcome"] = pd.factorize(df[found_name])[0]
                else:
                    # use coerced numeric values (NaNs remain if any)
                    df["outcome"] = coerced
    else:
        # No candidate name found. Try numeric columns.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Use the first numeric column as outcome (create 'outcome' copy).
            df["outcome"] = df[numeric_cols[0]].copy()
        else:
            # No numeric columns at all. Use the first column and factorize it.
            if len(df.columns) >= 1:
                first_col = df.columns[0]
                df["outcome"] = pd.factorize(df[first_col])[0]
            else:
                # Empty dataframe: create an empty numeric outcome column
                df["outcome"] = pd.Series(dtype=float)

    # Final safety: ensure 'outcome' is numeric. If not, factorize it.
    if not pd.api.types.is_numeric_dtype(df["outcome"]):
        df["outcome"] = pd.factorize(df["outcome"])[0]

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a simple statistical model to the transformed dataframe.

    Requirements:
    - df must contain an 'outcome' column (numeric). If not present, raises ValueError.

    Behavior:
    - Uses all other columns as predictors. Categorical predictors are turned into dummy variables.
    - If there are no predictors, fits an intercept-only model.
    - Attempts to use statsmodels. If statsmodels is unavailable, falls back to scikit-learn
      LinearRegression. If sklearn is also unavailable, falls back to numpy least squares.
    - Returns the fitted model object (statsmodels results if available, otherwise a fallback structure).
    """
    if "outcome" not in df.columns:
        outcome_candidates = ["outcome", "y", "target"]
        raise ValueError("No outcome column found. Expected one of: " + ", ".join(outcome_candidates))

    y = df["outcome"]

    # Prepare predictors: drop the outcome column
    X = df.drop(columns=["outcome"])

    # If there are no predictors, create a constant-only predictor
    if X.shape[1] == 0:
        X_proc = pd.DataFrame({"const": np.ones(len(df))})
    else:
        # Convert categorical variables to dummies, keep numeric as-is
        X_proc = pd.get_dummies(X, drop_first=True)
        # If get_dummies produces zero columns (e.g., all columns were empty), add constant
        if X_proc.shape[1] == 0:
            X_proc = pd.DataFrame({"const": np.ones(len(df))})

    # Ensure index alignment and no NA in predictors; if NA present, try to fill with column means
    if X_proc.isna().any().any():
        # For numeric columns: fill with mean
        numeric_cols = X_proc.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X_proc[col].isna().any():
                X_proc[col] = X_proc[col].fillna(X_proc[col].mean())
        # For any remaining non-numeric (unlikely after get_dummies), fill with 0
        X_proc = X_proc.fillna(0)

    # Ensure y is numeric
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]

    # Try statsmodels first
    try:
        import statsmodels.api as sm

        Xc = sm.add_constant(X_proc, has_constant="add")
        model_sm = sm.OLS(y, Xc).fit()
        return model_sm
    except Exception:
        # Fallback to scikit-learn
        try:
            from sklearn.linear_model import LinearRegression

            lr = LinearRegression()
            lr.fit(X_proc.values, np.asarray(y))
            return {"sklearn_model": lr, "X_columns": X_proc.columns.tolist()}
        except Exception:
            # Final fallback: use numpy lstsq to compute coefficients (including intercept)
            X_mat = np.asarray(X_proc, dtype=float)
            y_vec = np.asarray(y, dtype=float)
            # Add intercept column
            X_with_intercept = np.hstack([np.ones((X_mat.shape[0], 1)), X_mat])
            try:
                coef, *_ = np.linalg.lstsq(X_with_intercept, y_vec, rcond=None)
                return {"numpy_lstsq_coef": coef, "X_columns": ["const"] + X_proc.columns.tolist()}
            except Exception as e:
                # If everything fails, raise an informative error
                raise RuntimeError("Failed to fit model with available backends.") from e