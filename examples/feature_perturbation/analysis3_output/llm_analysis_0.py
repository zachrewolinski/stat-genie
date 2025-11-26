import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Any


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe into a form suitable for statistical modeling.

    Transformations performed:
    - Work on a copy of the dataframe to avoid side-effects.
    - Convert object-typed columns to categorical.
    - Fill numeric missing values with the column median (or 0 if median is NaN).
    - For categorical columns, add a 'missing' category and fill missing values with it.
    - Create derived features when common columns exist:
        - log_income (log of income if > 0, otherwise 0.0)
        - has_income (indicator for income not missing)
        - age_sq (square of age) and ensure age is non-negative
    - Normalize/convert 'treatment' into a binary 0/1 column when present.
    - Leaves other columns intact.
    """
    df = df.copy()

    # Convert object columns to categorical
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        try:
            df[col] = df[col].astype('category')
        except Exception:
            # If conversion fails for any reason, leave as-is
            pass

    # Fill numeric missing values with median (or 0 if median is NaN)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        med = df[col].median()
        if pd.isna(med):
            med = 0
        df[col] = df[col].fillna(med)

    # For categorical columns, add 'missing' and fillna with it
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    for col in cat_cols:
        try:
            if 'missing' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['missing'])
            df[col] = df[col].fillna('missing')
        except Exception:
            # If any unexpected issue, fallback to string fill
            df[col] = df[col].astype(str).fillna('missing').astype('category')

    # Derived features
    if 'income' in df.columns:
        # log_income: log(x) for x>0, else 0.0
        df['log_income'] = df['income'].apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else 0.0)
        df['has_income'] = df['income'].notna().astype(int)

    if 'age' in df.columns:
        # Ensure non-negative ages and create quadratic term
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
        df['age'] = df['age'].clip(lower=0)
        df['age_sq'] = df['age'] ** 2

    # Normalize/convert treatment to binary 0/1 if present
    if 'treatment' in df.columns:
        # If categorical, use codes (map -1 to 0)
        if df['treatment'].dtype.name == 'category':
            codes = df['treatment'].cat.codes.replace(-1, 0)
            # If codes have more than two distinct values, collapse to binary by >0
            if len(np.unique(codes)) > 2:
                df['treatment'] = (codes > 0).astype(int)
            else:
                df['treatment'] = codes.astype(int)
        else:
            # Try numeric conversion; non-numeric -> NaN -> treated as 0
            numeric = pd.to_numeric(df['treatment'], errors='coerce').fillna(0)
            # If values are not binary, map >0 to 1
            unique_vals = pd.Series(numeric).unique()
            if set(np.unique(unique_vals)).issubset({0, 1}):
                df['treatment'] = numeric.astype(int)
            else:
                df['treatment'] = (numeric > 0).astype(int)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear model (OLS) on the transformed dataframe.

    Behavior:
    - Attempts to find an outcome column in the following order:
        'outcome', 'y', 'Y', 'target'; if none found, uses the first numeric column.
    - Drops rows with missing outcome.
    - Uses all other columns except common id columns as predictors.
    - Numeric predictors are used as-is.
    - Categorical/object predictors are converted to dummy variables (drop_first=True).
    - Adds an intercept and fits OLS with robust HC3 covariance.
    - Returns the fitted results object (statsmodels RegressionResults).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input to model(...) must be a pandas DataFrame.")

    df = df.copy()

    # Determine outcome variable
    outcome_candidates = ['outcome', 'y', 'Y', 'target']
    outcome = None
    for c in outcome_candidates:
        if c in df.columns:
            outcome = c
            break

    if outcome is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("No suitable outcome column found (no numeric columns present).")
        outcome = num_cols[0]

    # Drop rows with missing outcome
    df = df.loc[df[outcome].notna()].copy()
    if df.shape[0] == 0:
        raise ValueError("No rows with non-missing outcome to fit the model.")

    # Exclude identifiers and outcome from predictors
    exclude = {outcome}
    for id_name in ['id', 'ID', 'Id']:
        if id_name in df.columns:
            exclude.add(id_name)

    predictor_cols = [c for c in df.columns if c not in exclude]
    if not predictor_cols:
        raise ValueError("No predictor columns available to fit the model.")

    # Split predictors into numeric and categorical/object
    X_num = df[predictor_cols].select_dtypes(include=[np.number]).copy()
    X_cat = df[predictor_cols].select_dtypes(include=['category', 'object']).copy()

    # If any categorical columns exist, get dummies (drop_first=True to avoid collinearity)
    if not X_cat.empty:
        X_cat = pd.get_dummies(X_cat, drop_first=True)

    # Combine numeric and dummy-coded categorical predictors
    if X_num.empty and X_cat.empty:
        raise ValueError("No valid predictors after encoding.")
    if X_num.empty:
        X = X_cat
    elif X_cat.empty:
        X = X_num
    else:
        X = pd.concat([X_num, X_cat], axis=1)

    # Ensure there are no duplicate column names and handle any all-NaN columns
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.dropna(axis=1, how='all')
    if X.shape[1] == 0:
        raise ValueError("No valid predictors remain after cleaning.")

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Prepare outcome vector
    y = pd.to_numeric(df[outcome], errors='coerce')
    if y.isna().all():
        raise ValueError("Outcome column could not be converted to numeric values for modeling.")
    # Align y with X in case of any differing indices
    y = y.loc[X.index]

    # Fit OLS with robust HC3 covariance
    model_result = sm.OLS(y, X).fit(cov_type='HC3')

    return model_result