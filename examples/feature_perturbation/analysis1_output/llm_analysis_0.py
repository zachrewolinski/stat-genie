import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Any


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for modeling by converting pandas extension dtypes
    (nullable integers, boolean extension dtypes, string extension dtypes, etc.)
    into NumPy-backed dtypes that patsy/statsmodels can interpret.

    This function:
    - Makes a shallow copy of the dataframe.
    - Converts pandas nullable integer dtypes (e.g. "Int64") to float64 to preserve NA.
    - Converts pandas boolean extension dtype ("boolean") to native bool (NA -> False).
    - Converts pandas string/categorical extension dtypes to object dtype (Python strings).
    - Leaves other columns unchanged.

    Returns the transformed dataframe. The returned dataframe contains the same
    columns as the input, but with safe dtypes for modeling.
    """
    df = df.copy()

    for col in df.columns:
        dtype = df[col].dtype

        # If it's an extension dtype (pandas nullable integer, boolean, string, etc.)
        if pd.api.types.is_extension_array_dtype(dtype):
            # Nullable integer -> convert to float64 to preserve NaNs (statsmodels accepts numeric floats)
            if pd.api.types.is_integer_dtype(dtype):
                df[col] = df[col].astype("float64")
            # Nullable boolean -> convert to native bool, fill NaN with False (explicit choice)
            elif pd.api.types.is_bool_dtype(dtype):
                # fillna(False) to avoid object dtype when converting
                df[col] = df[col].fillna(False).astype("bool")
            # String/objects/categorical -> convert to object (regular python strings) or categorical
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                # Convert categorical to object (patsy handles both, but object is safer for formula parsing)
                df[col] = df[col].astype("object")
            else:
                # Fallback: try to cast to float, otherwise to object
                try:
                    df[col] = df[col].astype("float64")
                except Exception:
                    df[col] = df[col].astype("object")

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a GLM to the provided (transformed) dataframe.

    This function:
    - Makes a defensive copy of the dataframe.
    - Ensures there are no pandas extension dtypes remaining (re-applies safe conversions).
    - Selects a numeric response variable (the first numeric column).
    - Uses all other columns as predictors.
    - Wraps object/string/categorical predictors with C() in the formula so they're treated as categorical.
    - Attempts to fit a Poisson GLM with robust HC3 covariance; if that fails, falls back to NegativeBinomial.

    Returns the fitted model results object.
    """
    df = df.copy()

    # Make sure no extension dtypes remain (repeat of transform's conversions, defensive)
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_extension_array_dtype(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                df[col] = df[col].astype("float64")
            elif pd.api.types.is_bool_dtype(dtype):
                df[col] = df[col].fillna(False).astype("bool")
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                df[col] = df[col].astype("object")
            else:
                try:
                    df[col] = df[col].astype("float64")
                except Exception:
                    df[col] = df[col].astype("object")

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in dataframe to use as a response for modeling.")

    # Choose the response variable.
    # Use the first numeric column as a pragmatic default.
    response = numeric_cols[0]

    # Build predictor list (all other columns)
    predictors = [c for c in df.columns if c != response]

    # Helper to format predictor terms: categorical-like -> C(col)
    def term(col_name: str) -> str:
        col_dtype = df[col_name].dtype
        if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_categorical_dtype(col_dtype) or pd.api.types.is_string_dtype(
            col_dtype
        ):
            return f"C({col_name})"
        else:
            return col_name

    if predictors:
        rhs = " + ".join(term(c) for c in predictors)
        formula = f"{response} ~ {rhs}"
    else:
        formula = f"{response} ~ 1"

    # Fit model: prefer Poisson with robust covariance; fall back to NegativeBinomial if necessary.
    try:
        results = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit(cov_type="HC3")
    except Exception:
        results = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()

    return results