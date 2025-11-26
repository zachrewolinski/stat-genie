import typing
from typing import Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for modeling.

    Ensures:
    - A numeric outcome column named 'y' exists (if possible, inferred from common names or first numeric column;
      otherwise a zero column is created).
    - At least one numeric predictor is present. Numeric predictors are taken from existing numeric columns except 'y'.
      If none exist, categorical columns are converted to dummies. If still none, a constant predictor is created.
    - Missing values in predictors and 'y' are filled with 0.0.
    - The list of predictor column names is stored in df.attrs['model_predictors'] for use by model().

    The function returns a copy of the input DataFrame with these guarantees.
    """
    # Make a safe copy
    df = pd.DataFrame(df).copy()

    # Find or create the target column 'y'
    y_candidates = [c for c in df.columns if str(c).lower() in ("y", "outcome", "target", "response", "label")]
    if y_candidates:
        df["y"] = pd.to_numeric(df[y_candidates[0]], errors="coerce")
    else:
        # Prefer first numeric column if available
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df["y"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
        else:
            # No numeric columns: create a zero outcome column
            df["y"] = 0.0

    # Identify numeric predictors excluding 'y'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors: List[str] = [c for c in numeric_cols if c != "y"]

    # If no numeric predictors, try converting categorical columns to dummies
    if not predictors:
        non_numeric_cols = [c for c in df.columns if c != "y"]
        if non_numeric_cols:
            dummies = pd.get_dummies(df[non_numeric_cols], drop_first=True)
            if not dummies.empty:
                # Attach dummy columns to df
                for col in dummies.columns:
                    df[col] = dummies[col].astype(float)
                predictors = dummies.columns.tolist()

    # If still no predictors, create a constant predictor
    if not predictors:
        df["const_feature"] = 1.0
        predictors = ["const_feature"]

    # Ensure y and predictors are numeric and fill NA with 0.0
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    for col in predictors:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Store predictor names in DataFrame attrs for the model function to pick up
    df.attrs["model_predictors"] = predictors

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model with robust (HC3) standard errors on the transformed DataFrame.

    Behavior:
    - Uses predictors from df.attrs['model_predictors'] when available.
    - Otherwise infers numeric predictors (all numeric cols except 'y').
    - Ensures there is at least one predictor (creates a constant predictor if necessary).
    - Drops rows with NA in y or predictors. If all rows are dropped, creates a single-row fallback dataset.
    - Returns the fitted statsmodels RegressionResultsWrapper.

    Note: The function expects the DataFrame to contain a numeric 'y' column.
    """
    # Defensive copy
    df = pd.DataFrame(df).copy()

    # Load predictor names from attrs if present
    preds = df.attrs.get("model_predictors", None)

    if preds is None:
        # Infer numeric predictors except 'y'
        preds = [c for c in df.select_dtypes(include=[np.number]).columns if c != "y"]
    else:
        # If stored as a string (older transform might store comma-separated), handle that
        if isinstance(preds, str):
            preds = [p for p in preds.split(",") if p]

    # Keep only predictors that actually exist in the DataFrame
    preds = [p for p in preds if p in df.columns]

    # If no valid predictors, create a constant predictor
    if not preds:
        df["const_feature"] = 1.0
        preds = ["const_feature"]

    # Prepare X and y
    X = df[preds].astype(float)
    # Add constant term if not already present
    X = sm.add_constant(X, has_constant="skip")
    if "y" not in df.columns:
        # Create a fallback y (zeros) if missing
        y = pd.Series(0.0, index=df.index, name="y")
    else:
        y = pd.to_numeric(df["y"], errors="coerce")

    # Combine and drop rows with NA in any used column
    data = pd.concat([y, X], axis=1)
    data = data.dropna()
    if data.shape[0] == 0:
        # Fallback single-row dataset to avoid zero-size arrays in statsmodels
        single_row = {col: 0.0 for col in X.columns}
        single_row["y"] = 0.0
        data = pd.DataFrame([single_row])

    y_clean = data["y"]
    X_clean = data.drop(columns=["y"])

    # Fit OLS with robust covariance (HC3)
    results = sm.OLS(y_clean, X_clean).fit(cov_type="HC3")

    return results