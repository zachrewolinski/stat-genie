import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Any


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe to ensure required modeling columns exist.

    - Ensures a numeric 'Deaths' column exists (by looking for common name variants).
    - Creates a 'LogDeaths' column = log(Deaths + 1) (uses 0 for missing deaths).
    - Returns a copy of the dataframe with the new/normalized columns added.

    This function is defensive: it will not raise if a deaths-like column is missing;
    instead it will create a 'Deaths' column filled with NaN and a 'LogDeaths' with
    log(0 + 1) = 0 where appropriate.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    df = df.copy()

    # Try to find a column that represents deaths (several common variants)
    deaths_col = None
    # exact match first
    for c in df.columns:
        if c.lower() == "deaths":
            deaths_col = c
            break
    # broader match if exact not found
    if deaths_col is None:
        for c in df.columns:
            if "death" in c.lower():
                deaths_col = c
                break

    # Create a normalized 'Deaths' column (numeric)
    if deaths_col is not None:
        df["Deaths"] = pd.to_numeric(df[deaths_col], errors="coerce")
    else:
        # If no deaths column exists, create it as NaN (so downstream logic can handle it)
        df["Deaths"] = np.nan

    # Create a log-transformed deaths column that is safe for zeros/missing values
    # Use fillna(0) so that missing values are treated as 0 for the log transform,
    # producing log(1) = 0. If you'd rather keep NaN, change fillna behavior.
    df["LogDeaths"] = np.log(df["Deaths"].fillna(0).astype(float) + 1)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a simple linear model predicting LogDeaths.

    - Ensures 'LogDeaths' exists (calls transform if necessary).
    - Searches for reasonable numeric predictor columns (population/case-like columns).
    - Always includes an intercept.
    - Returns the fitted statsmodels results object when fitting is possible.
    - If there is not enough data to fit a model, returns an informative dict.

    Notes:
    - This function is defensive and will not raise on typical missing-column issues.
    - The returned object is either a statsmodels.regression.linear_model.RegressionResultsWrapper
      or a dict describing why a model could not be fit.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Ensure transformation has been applied
    if "LogDeaths" not in df.columns:
        df = transform(df)
    else:
        # make a shallow copy to avoid modifying caller's df
        df = df.copy()

    # Identify candidate predictor columns (exclude target columns)
    excluded = {"logdeaths", "deaths"}
    predictor_candidates = []
    for c in df.columns:
        if c.lower() in excluded:
            continue
        # pick columns that look like useful numeric predictors:
        lower = c.lower()
        if any(key in lower for key in ("population", "pop", "case", "confirmed", "new", "age", "median")):
            predictor_candidates.append(c)

    # Coerce predictors to numeric where possible
    if predictor_candidates:
        exog = df[predictor_candidates].apply(pd.to_numeric, errors="coerce")
    else:
        # If no sensible predictors found, create an empty DataFrame to which we'll add intercept
        exog = pd.DataFrame(index=df.index)

    # Always include an intercept column
    exog = exog.copy()
    exog["Intercept"] = 1.0

    # Endogenous variable
    endog = pd.to_numeric(df["LogDeaths"], errors="coerce")

    # Combine and drop rows with NA in either endog or exog
    model_df = pd.concat([endog.rename("LogDeaths"), exog], axis=1)
    model_df = model_df.dropna()

    if model_df.shape[0] < 2:
        # Not enough observations to fit a regression
        return {
            "message": "Not enough complete observations to fit model",
            "n_observations_total": len(df),
            "n_observations_complete_cases": int(model_df.shape[0]),
        }

    y = model_df["LogDeaths"]
    X = model_df.drop(columns=["LogDeaths"])

    try:
        results = sm.OLS(y, X).fit()
    except Exception as e:
        # In case something unexpected goes wrong, return a descriptive dict
        return {"message": "Model fitting failed", "error": str(e)}

    return results