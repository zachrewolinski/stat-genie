import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Any


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for modeling:
    - Work on a copy to avoid mutating the input.
    - Drop columns that are completely empty.
    - For object or category dtypes, replace missing values with the literal 'missing'
      and convert them to pandas categorical dtype (so that patsy/statsmodels treat them
      as categorical predictors).
    - For boolean columns, fill missing with False.
    - Leave numeric missing values as-is (they will be handled by model-fitting which drops rows with missing response).
    Returns the transformed dataframe.
    """
    df = df.copy()

    # Drop columns that are entirely null
    df = df.dropna(axis=1, how="all")

    # Normalize object and category columns: fill NA and make them categorical
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].fillna("missing").astype("category")

    # Fill NA for boolean columns
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].fillna(False)

    # If any column has dtype 'category' but no categories (possible in some rare cases), ensure it has at least one
    for col in df.select_dtypes(include=["category"]).columns:
        if len(df[col].cat.categories) == 0:
            # Create a single category if no categories exist
            df[col] = df[col].cat.add_categories(["missing"]).fillna("missing")

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model using statsmodels.formula.api. The response is selected as follows:
    - If a column named 'y', 'outcome', or 'target' (case-insensitive) exists, the first such column is used.
    - Otherwise the first numeric column is used as the response.
    Predictors are all other columns except those that:
    - are entirely missing,
    - contain only un-hashable complex types (lists/dicts),
    - or have zero non-null unique values.
    Categorical predictors are passed to the formula as C(col).
    Columns with names that are not valid Python identifiers are renamed internally to safe names
    so the formula parsing does not fail.
    Returns the fitted statsmodels results object.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input to model must be a pandas DataFrame")

    model_df = df.copy()

    # Determine response column
    lower_cols = {c.lower(): c for c in model_df.columns}
    response = None
    for candidate in ("y", "outcome", "target"):
        if candidate in lower_cols:
            response = lower_cols[candidate]
            break

    if response is None:
        numeric_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No suitable numeric response column found and no 'y'/'outcome'/'target' column present.")
        response = numeric_cols[0]

    # Drop rows with missing response
    model_df = model_df.dropna(subset=[response])

    # Identify candidate predictors
    predictors = [c for c in model_df.columns if c != response]

    valid_predictors = []
    for col in predictors:
        series = model_df[col]
        # skip if all missing
        if series.dropna().shape[0] == 0:
            continue
        # skip columns that contain un-hashable container types in any row (lists/dicts)
        if series.dropna().apply(lambda x: isinstance(x, (list, dict, set))).any():
            continue
        # skip if zero non-null unique values
        if series.dropna().nunique() == 0:
            continue
        valid_predictors.append(col)

    if not valid_predictors:
        raise ValueError("No valid predictors available for modeling after filtering.")

    # Create a safe mapping of column names to valid Python identifiers to avoid patsy parsing issues
    all_used_cols = [response] + valid_predictors
    safe_mapping = {}
    used_safe_names = set()
    for idx, col in enumerate(all_used_cols):
        # prefer original name if it's a valid identifier
        if isinstance(col, str) and col.isidentifier():
            safe = col
        else:
            safe = f"col_{idx}"
        # ensure uniqueness
        base = safe
        i = 1
        while safe in used_safe_names:
            safe = f"{base}_{i}"
            i += 1
        safe_mapping[col] = safe
        used_safe_names.add(safe)

    # Rename dataframe for modeling
    model_df_safe = model_df.rename(columns=safe_mapping)

    response_safe = safe_mapping[response]
    predictor_safes = [safe_mapping[c] for c in valid_predictors]

    # Build formula terms, treating categorical predictors explicitly with C(...)
    terms = []
    for orig_col, safe_col in zip(valid_predictors, predictor_safes):
        orig_series = model_df[orig_col]
        if pd.api.types.is_categorical_dtype(orig_series) or orig_series.dtype == object:
            # ensure there is at least one non-null category
            if orig_series.dropna().nunique() == 0:
                continue
            terms.append(f"C({safe_col})")
        else:
            terms.append(safe_col)

    if not terms:
        raise ValueError("No usable predictor terms available to build formula.")

    ols_formula = f"{response_safe} ~ " + " + ".join(terms)

    # Fit OLS model; statsmodels will drop any rows with missing predictor values automatically
    results = smf.ols(formula=ols_formula, data=model_df_safe).fit()

    return results