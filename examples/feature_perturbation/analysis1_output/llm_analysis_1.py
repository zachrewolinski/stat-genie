import pandas as pd
import numpy as np
from typing import Any
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for modeling by converting pandas extension dtypes
    (e.g. nullable integer "Int64", nullable boolean "boolean", etc.)
    to numpy-backed dtypes that statsmodels/patsy can consume.

    The function:
    - Makes and returns a copy of the dataframe (does not modify in place).
    - Converts pandas extension integer/float dtypes to numpy float64.
      (float64 is used to preserve NA values).
    - Converts pandas nullable boolean dtype to numpy bool (NA will be converted to False).
    - Converts other extension types (including object/categorical) to types
      that are compatible; categorical is left as-is because downstream
      encoding uses get_dummies.
    """
    df = df.copy()

    # pandas API helpers
    from pandas.api import types as ptypes

    for col in df.columns:
        dtype = df[col].dtype

        # Skip already numpy-backed numeric dtypes
        if np.issubdtype(dtype, np.number):
            continue

        # If it's a pandas extension array dtype (e.g., "Int64", "Float64", "boolean")
        if ptypes.is_extension_array_dtype(dtype):
            # Nullable integers or floats -> convert to numpy float64 (keeps NaN)
            if ptypes.is_integer_dtype(dtype) or ptypes.is_float_dtype(dtype):
                df[col] = df[col].astype("float64")
            # Nullable booleans -> convert to numpy bool, fill NA with False
            elif ptypes.is_bool_dtype(dtype):
                # fillna(False) to ensure conversion to numpy bool works
                df[col] = df[col].fillna(False).astype("bool")
            # CategoricalDtype is fine to leave as category; objects remain as-is
            else:
                # fallback: convert to object for safety (string-like)
                try:
                    df[col] = df[col].astype("object")
                except Exception:
                    # as a last resort, convert via numpy array
                    df[col] = df[col].to_numpy()

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial GLM to the dataframe without using Patsy/formula,
    to avoid issues with pandas extension dtypes and Patsy dtype sniffing.

    Strategy:
    - Automatically pick a response (endogenous) variable: the first numeric column found.
    - Use all other columns as predictors.
    - For categorical/object/bool predictors, use one-hot encoding (pd.get_dummies).
    - For numeric predictors, use them as-is.
    - Add a constant (intercept) and fit statsmodels.GLM with NegativeBinomial family.

    Returns the fitted results object (statsmodels RegressionResultsWrapper).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Work on a copy to avoid modifying original
    data = df.copy()

    # Select numeric columns for candidate response
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to select as response variable.")

    # Choose the first numeric column as response
    y_col = numeric_cols[0]

    # Features: all other columns
    feature_cols = [c for c in data.columns if c != y_col]
    if len(feature_cols) == 0:
        # If there are no predictors, fit intercept-only model
        y = data[y_col].astype(float)
        X = pd.DataFrame({"const": np.ones(len(y))}, index=data.index)
    else:
        X_parts = []
        for col in feature_cols:
            col_series = data[col]

            # If the column is numeric, use as-is (convert to float to handle NaNs)
            if np.issubdtype(col_series.dtype, np.number):
                X_parts.append(col_series.astype(float))
            else:
                # For non-numeric (object/category/bool), use one-hot encoding
                # drop_first=True to avoid multicollinearity where possible
                dummies = pd.get_dummies(col_series, prefix=col, drop_first=True, dummy_na=False)
                if not dummies.empty:
                    X_parts.append(dummies)

        if not X_parts:
            # No usable predictors -- intercept-only model
            X = pd.DataFrame({"const": np.ones(len(data))}, index=data.index)
        else:
            # Concatenate all parts into a single design matrix
            X = pd.concat(X_parts, axis=1)
            # Ensure all columns are numeric dtype
            for c in X.columns:
                if not np.issubdtype(X[c].dtype, np.number):
                    X[c] = pd.to_numeric(X[c], errors="coerce")

            # Add intercept
            X = sm.add_constant(X, has_constant="add")

        y = data[y_col].astype(float)

    # Align and drop rows with missing values in X or y
    combined = pd.concat([y, X], axis=1)
    combined = combined.dropna()
    if combined.shape[0] == 0:
        raise ValueError("No rows available after dropping missing values for model fitting.")

    y_clean = combined.iloc[:, 0]
    X_clean = combined.iloc[:, 1:]

    # Fit Negative Binomial GLM
    model = sm.GLM(y_clean, X_clean, family=sm.families.NegativeBinomial())
    results = model.fit()

    return results