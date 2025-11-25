import pandas as pd
import numpy as np
from typing import Any, List, Optional
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe to prepare features for statistical modeling.

    Transformations performed:
    - Works on a copy of the input dataframe; original dataframe is not modified.
    - Adds 'num_missing' column: count of missing values per row.
    - For each numeric column with missing values, creates a new column
      '<col>_imputed' where missing values are filled with the column median.
    - For each non-numeric (object/category) column, creates a new column
      '<col>_filled' where missing values are replaced with the string 'missing',
      and then creates one-hot encoded indicator columns (prefix '<col>') for that
      filled column, dropping the first level to avoid perfect multicollinearity.
    - Leaves original columns intact (as requested: changes/additions are new columns).

    The returned dataframe includes the original columns plus the newly derived
    columns needed for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Original input dataframe.

    Returns
    -------
    pd.DataFrame
        Transformed dataframe containing original and newly derived columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df_out = df.copy(deep=True)

    # Add count of missing values per row
    df_out["num_missing"] = df_out.isna().sum(axis=1)

    # Process numeric columns: impute missing values into new columns
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df_out[col].isna().any():
            imputed_col = f"{col}_imputed"
            median_val = df_out[col].median(skipna=True)
            # If median is NaN (all values missing), fill with 0 to ensure numeric column
            if np.isnan(median_val):
                median_val = 0.0
            df_out[imputed_col] = df_out[col].fillna(median_val)

    # Process non-numeric columns: fill missing and create dummies
    non_numeric_cols = df_out.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in non_numeric_cols:
        filled_col = f"{col}_filled"
        # Fill missing with explicit token
        df_out[filled_col] = df_out[col].fillna("missing").astype("category")
        # Create dummies for modeling, drop first to reduce multicollinearity
        dummies = pd.get_dummies(df_out[filled_col], prefix=col, drop_first=True, dtype=int)
        if not dummies.empty:
            # Only add dummies if there is at least one level besides the dropped one
            df_out = pd.concat([df_out, dummies], axis=1)

    # Ensure column names are safe (no duplicates introduced by transformations)
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    return df_out


def model(df: pd.DataFrame) -> Any:
    """
    Fit a simple linear regression model (OLS) using numeric predictors from the
    transformed dataframe and return the fitted results object.

    Behavior:
    - Attempts to locate an outcome column in this order: 'y', 'target', 'outcome'.
      If none are present, the first numeric column (left-to-right) will be used
      as the outcome.
    - Predictors are all numeric columns except the chosen outcome column.
      Missing predictor values are filled with 0.
    - Adds an intercept term automatically.
    - Returns the fitted statsmodels regression results object.

    Parameters
    ----------
    df : pd.DataFrame
        Transformed dataframe produced by transform().

    Returns
    -------
    Any
        The fitted statsmodels results object (e.g., RegressionResultsWrapper).

    Raises
    ------
    ValueError
        If no suitable outcome or predictor columns can be found.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df_in = df.copy(deep=True)

    # Determine outcome column
    possible_outcomes = ["y", "target", "outcome"]
    outcome_col: Optional[str] = None
    for name in possible_outcomes:
        if name in df_in.columns:
            outcome_col = name
            break

    if outcome_col is None:
        # pick first numeric column as outcome
        numeric_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns available to select as outcome.")
        outcome_col = numeric_cols[0]

    # Drop rows with missing outcome
    df_in = df_in.dropna(subset=[outcome_col])
    y = df_in[outcome_col].astype(float)

    # Select numeric predictors excluding the outcome column
    predictor_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    predictor_cols = [c for c in predictor_cols if c != outcome_col]

    if not predictor_cols:
        raise ValueError("No numeric predictor columns available for modeling.")

    X = df_in[predictor_cols].fillna(0.0).astype(float)

    # Add intercept
    X_with_const = sm.add_constant(X, has_constant="add")

    # Fit OLS model
    model_fit = sm.OLS(y, X_with_const)
    results = model_fit.fit()

    return results