import pandas as pd
import numpy as np
from typing import Any, Optional

# Try to import statsmodels; if not available, we'll fall back to numpy OLS
try:
    import statsmodels.api as sm  # type: ignore
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe to ensure it contains the columns necessary for modeling.

    Transformations performed:
    - Ensure a 'Deaths' column exists. If the incoming dataframe uses a common alternative
      column name (e.g., 'deaths', 'death', 'Fatalities', 'fatalities', 'num_deaths'),
      that column will be copied/renamed to 'Deaths'. If no suitable column is found,
      a 'Deaths' column of zeros will be created.
    - Create 'LogDeathsPlus1' = log(Deaths.clip(lower=0) + 1).
    - Ensure an 'Intercept' column (all ones) exists to allow modeling with an intercept.
    - Leave all other columns intact; do not remove or overwrite existing columns
      except to add the standardized 'Deaths' column if needed.

    The function returns a new dataframe (a copy) and does not mutate the input.
    """
    df = df.copy()

    # Standardize possible death-like column names to 'Deaths'
    death_column_candidates = {
        "deaths",
        "death",
        "fatalities",
        "fatality",
        "num_deaths",
        "numdeaths",
        "deaths_count",
        "Deaths",
        "Deaths_Count",
        "DEATHS",
    }

    found_col: Optional[str] = None
    # First check exact column names (case-sensitive)
    for cand in death_column_candidates:
        if cand in df.columns:
            found_col = cand
            break

    # If not found, try case-insensitive match
    if found_col is None:
        cols_lower_map = {c.lower(): c for c in df.columns}
        for cand in death_column_candidates:
            lower = cand.lower()
            if lower in cols_lower_map:
                found_col = cols_lower_map[lower]
                break

    # Create or normalize 'Deaths' column
    if found_col is not None and found_col != "Deaths":
        # copy/convert to numeric, coerce errors to NaN then fill with 0
        df["Deaths"] = pd.to_numeric(df[found_col], errors="coerce").fillna(0.0)
    elif found_col == "Deaths":
        df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce").fillna(0.0)
    else:
        # No suitable column found — create a zero-filled Deaths column
        df["Deaths"] = pd.Series(0.0, index=df.index, dtype=float)

    # Create log-transformed target used by many models (log(x+1))
    # Clip at 0 to avoid negatives
    df["LogDeathsPlus1"] = np.log(df["Deaths"].clip(lower=0.0) + 1.0)

    # Add an intercept column for modeling convenience
    df["Intercept"] = 1.0

    return df


def _numpy_ols(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Simple OLS regression using numpy for environments without statsmodels.
    Returns a dictionary with params and residuals and basic summary info.
    """
    # Solve for beta in least squares sense
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # If residuals array is empty (happens when overdetermined differently),
    # compute residuals manually
    if residuals.size == 0:
        fitted = X @ beta
        residuals = np.array([np.sum((y - fitted) ** 2)])

    return {
        "params": beta,
        "residuals": residuals,
        "rank": rank,
        "singular_values": s,
    }


def model(df: pd.DataFrame) -> Any:
    """
    Fit a statistical model on the transformed dataframe.

    Behavior:
    - Uses LogDeathsPlus1 as the target variable.
    - Selects numeric predictors from the dataframe excluding 'Deaths' and 'LogDeathsPlus1'.
    - If statsmodels is available, fits an OLS model using statsmodels and returns the
      fitted results object.
    - Otherwise, falls back to a lightweight numpy least-squares fit and returns a dict
      containing parameter estimates and diagnostic information.

    The function will drop rows with missing values in predictors or the target.
    """
    if "LogDeathsPlus1" not in df.columns:
        raise ValueError("Transformed dataframe must contain 'LogDeathsPlus1'. Run transform(df) first.")

    # Select numeric predictors
    numeric = df.select_dtypes(include=[np.number]).copy()

    # Exclude the target and the raw 'Deaths' column from predictors
    if "LogDeathsPlus1" in numeric.columns:
        numeric = numeric.drop(columns=["LogDeathsPlus1"])
    if "Deaths" in numeric.columns:
        # keep 'Deaths' out of predictors unless explicitly desired — comment out if needed
        numeric = numeric.drop(columns=["Deaths"])

    # Ensure at least an intercept is present
    if "Intercept" not in numeric.columns:
        numeric["Intercept"] = 1.0

    y = df["LogDeathsPlus1"]

    # Align indices and drop rows with NA in predictors/target
    combined = pd.concat([numeric, y], axis=1)
    combined = combined.dropna()
    if combined.shape[0] == 0:
        raise ValueError("No data left after dropping NA rows. Cannot fit model.")

    X = combined[numeric.columns].to_numpy(dtype=float)
    y_clean = combined["LogDeathsPlus1"].to_numpy(dtype=float)

    # If statsmodels is available, use it for a richer result object
    if _HAS_STATSMODELS:
        # statsmodels typically expects an intercept or a constant column if desired;
        # since we include 'Intercept' we will not add another constant.
        model_sm = sm.OLS(y_clean, X)
        results = model_sm.fit()
        return results

    # Fallback to numpy lstsq
    results = _numpy_ols(X, y_clean)
    return results