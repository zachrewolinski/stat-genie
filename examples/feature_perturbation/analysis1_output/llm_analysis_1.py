import pandas as pd
import numpy as np
from typing import Any
import warnings
import statsmodels.api as sm
from pandas.api import types as pdt


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe into a form suitable for statistical modeling.

    Transformations performed:
    - Work on a copy of the dataframe (original is not modified).
    - Convert boolean columns to integers (0/1).
    - Convert datetime-like columns to numeric timestamps (seconds since epoch).
    - Convert object or categorical columns to dummy/indicator columns (one-hot encoding).
      Columns that look like datetimes (when parsed) will be treated as datetimes instead.
    - Numeric columns are left as-is, and missing numeric values are imputed with the column mean.
    - Missing values in categorical/dummy columns are filled with 0 (after one-hot encoding).
    - Add indicator columns for missingness for original columns (prefix "missing_").
    - The returned dataframe contains only numeric columns (suitable for most statistical models).

    Args:
        df: Input pandas DataFrame.

    Returns:
        Transformed pandas DataFrame with numeric columns only.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df = df.copy(deep=True)
    # Keep track of missingness indicators to add later
    missing_indicators = {}

    for col in df.columns:
        missing_indicators[f"missing_{col}"] = df[col].isna().astype(int)

    # We'll build a new dataframe of processed columns
    processed = {}

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        # Numeric types: keep, but coerce to numeric if object-like numeric strings
        if pdt.is_numeric_dtype(dtype):
            # Coerce to numeric to handle numeric strings; errors -> NaN
            coerced = pd.to_numeric(series, errors="coerce")
            processed[col] = coerced
            continue

        # Boolean -> convert to integer (0/1)
        if pdt.is_bool_dtype(dtype):
            processed[col] = series.astype(int)
            continue

        # If object or categorical, check whether it is datetime-like
        if pdt.is_object_dtype(dtype) or pdt.is_categorical_dtype(dtype):
            # Try parsing as datetime
            parsed = pd.to_datetime(series, errors="coerce")
            non_na_fraction = parsed.notna().sum() / max(1, len(parsed))
            # If a reasonable fraction parsed as datetime, treat as datetime
            if non_na_fraction > 0.5:
                # Convert to unix timestamp in seconds (float)
                ts = parsed.view("int64") // 10**9
                processed[col] = ts.astype("float64")
            else:
                # Treat as categorical: one-hot encode
                # First convert to string to ensure consistent dummies for mixed types
                cat_series = series.astype("category")
                dummies = pd.get_dummies(cat_series, prefix=col, dummy_na=False, drop_first=True)
                # Add each dummy column to processed
                for dummy_col in dummies.columns:
                    processed[dummy_col] = dummies[dummy_col].astype(float)
            continue

        # Datetime dtypes
        if pdt.is_datetime64_any_dtype(dtype):
            ts = series.view("int64") // 10**9
            processed[col] = ts.astype("float64")
            continue

        # Fallback: try to coerce to numeric
        try:
            coerced = pd.to_numeric(series, errors="coerce")
            processed[col] = coerced
        except Exception:
            # As a last resort, convert to string and one-hot encode (may be many columns)
            cat_series = series.astype("category")
            dummies = pd.get_dummies(cat_series, prefix=col, dummy_na=False, drop_first=True)
            for dummy_col in dummies.columns:
                processed[dummy_col] = dummies[dummy_col].astype(float)

    # Build dataframe
    processed_df = pd.DataFrame(processed, index=df.index)

    # Add missing indicators
    for ind_col, ind_series in missing_indicators.items():
        processed_df[ind_col] = ind_series.astype(int)

    # Impute numeric missing values with column mean
    for col in processed_df.columns:
        if pdt.is_numeric_dtype(processed_df[col].dtype):
            if processed_df[col].isna().any():
                mean_val = processed_df[col].mean()
                # If mean is nan (all values missing), fill with 0
                if pd.isna(mean_val):
                    mean_val = 0.0
                processed_df[col] = processed_df[col].fillna(mean_val)

    # Ensure all columns are numeric and of float64 type (common for modeling)
    for col in processed_df.columns:
        if not pdt.is_numeric_dtype(processed_df[col].dtype):
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce").fillna(0.0)
        processed_df[col] = processed_df[col].astype("float64")

    return processed_df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a simple linear regression model (OLS) using statsmodels if a target column is present.

    The function will look for a target column with a common name among:
    ['target', 'y', 'label', 'outcome'] (case-insensitive). If multiple are present, the first
    match in that list is used. If no target column is found, returns a dictionary summarizing
    the numeric columns (no exception is raised).

    Args:
        df: Transformed dataframe produced by transform(), should be numeric.

    Returns:
        If a target column is found: the fitted statsmodels regression results object.
        Otherwise: a dict with a summary of numeric columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.shape[0] == 0:
        raise ValueError("DataFrame is empty")

    # Identify target column
    target_candidates = ["target", "y", "label", "outcome"]
    lower_cols = {c.lower(): c for c in df.columns}
    target_col = None
    for cand in target_candidates:
        if cand in lower_cols:
            target_col = lower_cols[cand]
            break

    # If target not found, try to find any column named exactly 'response' or 'dep_var'
    if target_col is None:
        for cand in ["response", "dep_var"]:
            if cand in lower_cols:
                target_col = lower_cols[cand]
                break

    # If still not found, return a summary rather than raising
    if target_col is None:
        warnings.warn(
            "No target column found among common names. Returning dataframe numeric summary instead of fitting a model."
        )
        numeric_cols = [c for c in df.columns if pdt.is_numeric_dtype(df[c].dtype)]
        summary = {
            "n_rows": df.shape[0],
            "n_columns": df.shape[1],
            "numeric_columns": numeric_cols,
            "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        }
        return summary

    # Prepare X and y
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])

    # Ensure X contains at least one column
    if X.shape[1] == 0:
        raise ValueError(f"No predictor columns available after removing target '{target_col}'")

    # Use only numeric predictors
    numeric_predictors = [c for c in X.columns if pdt.is_numeric_dtype(X[c].dtype)]
    if len(numeric_predictors) == 0:
        raise ValueError("No numeric predictor columns available for modeling")

    X = X[numeric_predictors].astype(float)

    # Add constant for intercept
    X_with_const = sm.add_constant(X, has_constant="add")

    # Fit OLS regression
    try:
        model_res = sm.OLS(y, X_with_const, missing="drop").fit()
        return model_res
    except Exception as e:
        # In case statsmodels fails for some reason, provide informative error
        raise RuntimeError(f"Failed to fit OLS model: {e}") from e