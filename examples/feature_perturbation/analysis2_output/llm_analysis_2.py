import pandas as pd
import numpy as np
from typing import Any, Dict


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe to ensure it contains the columns needed for downstream modeling.

    Transformations performed:
    - Work on a copy of the input dataframe to avoid side-effects.
    - Locate a column that corresponds to deaths (case-insensitive, tolerant to small variations).
      The canonical column produced is 'Deaths'.
    - Coerce the 'Deaths' column to numeric (non-convertible values -> NaN).
    - Add an indicator 'Deaths_is_missing' that flags rows where 'Deaths' is missing.
    - Add a derived column 'log_Deaths' = log1p(Deaths_filled_with_0) for use in models that prefer
      a transformed response.
    - Leave all other columns intact.

    This function will not raise if a deaths-like column cannot be found; instead it will create
    a 'Deaths' column filled with NaN and appropriate derived columns so downstream code can handle it.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df_out = df.copy()

    # Try to find a column corresponding to deaths in a tolerant manner
    deaths_col = None
    for col in df_out.columns:
        col_stripped = str(col).strip()
        lower = col_stripped.lower()
        # Accept exact 'deaths' or variations that include the word 'death'
        if lower == "deaths" or lower == "death" or "death" in lower:
            deaths_col = col
            break

    if deaths_col is not None:
        # Create canonical 'Deaths' column, coercing to numeric
        df_out["Deaths"] = pd.to_numeric(df_out[deaths_col], errors="coerce")
    else:
        # If no suitable column found, create 'Deaths' as NaN so downstream can handle it
        df_out["Deaths"] = np.nan

    # Indicator for missing Deaths
    df_out["Deaths_is_missing"] = df_out["Deaths"].isna()

    # Derived transformation: log(Deaths + 1) with NaN -> treat as 0 for the transform to avoid -inf
    # We intentionally do not overwrite the original 'Deaths'; we add 'log_Deaths'.
    # Use fillna(0) so rows with missing deaths get log_Deaths = 0 (this is a modeling choice;
    # downstream modeling code can use Deaths_is_missing if needed).
    df_out["log_Deaths"] = np.log1p(df_out["Deaths"].fillna(0))

    return df_out


def model(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fit a simple linear model using numeric predictors in `df` to predict 'Deaths'.

    Behavior:
    - Response variable: attempts to use 'Deaths' if present; otherwise uses 'log_Deaths' if present;
      if neither exist, uses a zero-vector (no-information) response.
    - Predictors: all numeric columns except 'Deaths' and 'log_Deaths' are used as features.
      An intercept is always included.
    - Rows with NaN in the response or predictors are dropped for the fit.
    - The model is fitted via numpy.linalg.lstsq (ordinary least squares).
    - Returns a dictionary containing coefficients (including intercept), R-squared, predictions,
      residuals, and n_obs used in the fit.

    The function is written to avoid raising under normal input circumstances; if no usable rows
    exist for fitting, it returns coefficients for an intercept-only model predicting the mean (or 0).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Determine response vector y
    if "Deaths" in df.columns:
        y = df["Deaths"].copy()
    elif "log_Deaths" in df.columns:
        y = df["log_Deaths"].copy()
    else:
        # As a last resort, create a zero response to keep the function non-failing
        y = pd.Series(np.zeros(len(df)), index=df.index, name="Deaths")

    # Identify numeric predictors, excluding any response-like columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if c not in {"Deaths", "log_Deaths"}]

    # Build design matrix X (include intercept)
    if len(predictors) == 0:
        # Intercept-only model
        X = pd.DataFrame({"intercept": np.ones(len(df))}, index=df.index)
    else:
        X = df[predictors].copy()
        # Add intercept column
        X.insert(0, "intercept", 1.0)

    # Align y and X and drop rows with NaNs in either
    combined = pd.concat([y.rename("y"), X], axis=1)
    combined_clean = combined.dropna(axis=0, how="any")

    if combined_clean.shape[0] == 0:
        # No rows to fit: return sensible defaults
        # Intercept is mean of y if available, otherwise 0
        if y.dropna().shape[0] > 0:
            intercept_val = float(y.dropna().mean())
        else:
            intercept_val = 0.0

        coeffs = {"intercept": intercept_val}
        for col in predictors:
            coeffs[col] = 0.0

        n_obs = 0
        predictions = pd.Series(np.full(len(df), intercept_val), index=df.index, name="predicted")
        residuals = pd.Series(np.full(len(df), np.nan), index=df.index, name="residuals")

        results = {
            "coefficients": coeffs,
            "r_squared": float("nan"),
            "predictions": predictions,
            "residuals": residuals,
            "n_obs": n_obs,
        }
        return results

    # Proceed with ordinary least squares using numpy
    y_clean = combined_clean["y"].values
    X_clean = combined_clean.drop(columns="y").values
    # Solve least squares
    try:
        betas, residuals_sum, rank, s = np.linalg.lstsq(X_clean, y_clean, rcond=None)
    except np.linalg.LinAlgError:
        # In case of numerical issues, fall back to zeros
        betas = np.zeros(X_clean.shape[1])

    # Map coefficients back to names
    coef_names = combined_clean.drop(columns="y").columns.tolist()
    coefficients = {name: float(b) for name, b in zip(coef_names, betas)}

    # Predictions for all rows: use available columns in X (for rows with NaN predictors, result will be NaN)
    # Build full X matrix matching coef_names
    X_full = X[coef_names].values
    preds_full = X_full.dot(betas)
    predictions = pd.Series(preds_full, index=df.index, name="predicted")

    # Residuals: defined where both y and prediction are finite; otherwise NaN
    residuals = pd.Series(np.nan, index=df.index, name="residuals")
    common_index = combined_clean.index
    residuals.loc[common_index] = combined_clean["y"].values - (X_clean.dot(betas))

    # Compute R-squared on the cleaned data used for fitting
    ss_res = np.sum((combined_clean["y"].values - X_clean.dot(betas)) ** 2)
    ss_tot = np.sum((combined_clean["y"].values - combined_clean["y"].values.mean()) ** 2)
    if ss_tot > 0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = float("nan")  # undefined when y is constant

    results = {
        "coefficients": coefficients,
        "r_squared": float(r_squared),
        "predictions": predictions,
        "residuals": residuals,
        "n_obs": int(combined_clean.shape[0]),
    }
    return results