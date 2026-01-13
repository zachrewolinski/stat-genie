from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/add_features_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling fish caught per hour.
    Produces/keeps the following columns required by the model and analysis:
      - fish_caught (int)
      - livebait (int, 0/1)
      - camper (int, 0/1)
      - persons (int, original adults count)
      - child (int, original children count)
      - group_size (persons + child)
      - hours (float, clipped to a small positive lower bound)
      - log_hours (float, natural log of hours) used as offset in GLM
      - fish_per_hour (float, descriptive: fish_caught / hours)

    Drops rows with missing essential values.
    """
    # work on a copy
    df = df.copy()

    # Drop rows with missing critical fields
    df = df.dropna(subset=["fish_caught", "hours"])

    # Ensure numeric types for key columns
    df["fish_caught"] = pd.to_numeric(df["fish_caught"], errors="coerce")
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    df["persons"] = pd.to_numeric(df.get("persons", pd.Series(dtype=float)), errors="coerce")
    df["child"] = pd.to_numeric(df.get("child", pd.Series(dtype=float)), errors="coerce")

    # Drop rows that became NA after coercion
    df = df.dropna(subset=["fish_caught", "hours"]) 

    # Some very small or zero hours can cause problems with log(offset). Clip to a small positive value.
    min_hours = 1e-3
    df.loc[:, "hours"] = df["hours"].clip(lower=min_hours)

    # Create log_hours for offset
    df["log_hours"] = np.log(df["hours"])

    # Ensure binary predictors are integers 0/1
    if "livebait" in df.columns:
        df["livebait"] = pd.to_numeric(df["livebait"], errors="coerce").fillna(0).astype(int)
    else:
        df["livebait"] = 0

    if "camper" in df.columns:
        df["camper"] = pd.to_numeric(df["camper"], errors="coerce").fillna(0).astype(int)
    else:
        df["camper"] = 0

    # Fill persons/child missing values with 0 where appropriate (conservative) and make integer
    df["persons"] = df["persons"].fillna(0).astype(int)
    df["child"] = df["child"].fillna(0).astype(int)

    # Group size: total people in the party
    df["group_size"] = (df["persons"] + df["child"]).astype(int)

    # Descriptive rate per hour (useful for summaries and plotting)
    df["fish_per_hour"] = df["fish_caught"] / df["hours"]

    # Make fish_caught integer (count)
    df["fish_caught"] = df["fish_caught"].round().astype(int)

    # Keep only columns necessary for modeling and inspection
    cols_to_keep = [
        "fish_caught",
        "livebait",
        "camper",
        "persons",
        "child",
        "group_size",
        "hours",
        "log_hours",
        "fish_per_hour"
    ]

    # Some of these may not exist in edge cases, so intersect with available cols
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_to_keep]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fits a generalized linear model for count data using a log link and an offset for log(hours).
    Primary model: Negative Binomial GLM to allow for overdispersion relative to Poisson.

    Formula (in matrix form):
      E[fish_caught] = hours * exp(beta0 + beta1*livebait + beta2*camper + beta3*group_size)
    which is implemented as GLM with offset = log_hours and family = NegativeBinomial.

    Returns a dict with:
      - 'glm_results': the fitted statsmodels GLMResults object
      - 'predicted_count': the model's predicted expected counts per row
      - 'predicted_rate_per_hour': predicted_count / hours -> expected fish per hour per group
    """
    # Required columns for the model
    required = ["fish_caught", "livebait", "camper", "group_size", "hours", "log_hours"]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Build design matrix (exogenous variables)
    exog_vars = ["livebait", "camper", "group_size"]
    exog = df[exog_vars].copy()
    exog = sm.add_constant(exog, has_constant='add')

    # Endogenous
    endog = df["fish_caught"]

    # Fit Negative Binomial GLM with offset = log_hours
    try:
        model_glm = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(), offset=df["log_hours"])
        results = model_glm.fit()
    except Exception:
        # Fallback to Poisson if NegativeBinomial fails
        model_glm = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=df["log_hours"])
        results = model_glm.fit()

    # Predictions: expected count (mu) and convert to rate per hour
    predicted_count = results.predict(exog, offset=df["log_hours"])  # expected fish count for the trip
    predicted_rate_per_hour = predicted_count / df["hours"]

    # Attach predictions to a small dataframe for return/inspection
    preds = df[["hours"]].copy()
    preds["predicted_count"] = predicted_count
    preds["predicted_rate_per_hour"] = predicted_rate_per_hour

    return {
        "glm_results": results,
        "predictions": preds
    }


