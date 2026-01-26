from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into the analysis dataframe.

    Creates the following columns used in the modeling step:
      - name_mf: raw continuous masculinity-femininity score (higher = more feminine) taken from column 'name'.
      - name_mf_z: z-scored version of name_mf for interpretability.
      - female_name: binary indicator from 'elapsedyrs' (expects 0=male, 1=female per dataset description).
      - ndam15: raw death counts (keeps original column for GLM).
      - log_deaths: log(1 + ndam15) used as the primary dependent variable for OLS.
      - saffir_category: from 'masfem' (Saffir-Simpson category) as numeric severity control.
      - max_wind: from 'wind'.
      - min_pressure: from 'min'.
      - ind: original property damage variable (keeps raw for checks).
      - log_property_damage: log(1 + ind) to reduce skew.
      - year: year of hurricane taken from 'alldeaths' (the dataset field documented as year).

    The function will drop rows missing the essential variables for modeling.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Select and rename the key columns as they appear in the raw dataset
    # Note: according to the schema, 'name' is the coder-average masculinity-femininity index
    # 'elapsedyrs' is the binary gender indicator (0 male, 1 female)
    # 'ndam15' is the total number of deaths
    # 'masfem' is the Saffir-Simpson category
    # 'wind' is max wind speed, 'min' is minimum pressure, 'ind' is property damage (normalized)
    # 'alldeaths' is described as the year variable in the provided schema

    # Ensure these expected columns exist; if not, code will raise a KeyError which should alert the user

    # Drop rows missing essential variables used by the main model
    essential_cols = ["name", "elapsedyrs", "ndam15", "masfem", "wind", "min", "ind", "alldeaths"]
    df = df.dropna(subset=essential_cols)

    # Create analysis columns
    # 1) femininity score and z-score
    df["name_mf"] = pd.to_numeric(df["name"], errors="coerce")
    # If any coercion produced NaNs, drop them
    df = df.dropna(subset=["name_mf"])
    df["name_mf_z"] = (df["name_mf"] - df["name_mf"].mean()) / (df["name_mf"].std(ddof=0) if df["name_mf"].std(ddof=0) != 0 else 1.0)

    # 2) binary female name indicator
    # according to schema: elapsedyrs is 0 for male, 1 for female
    df["female_name"] = pd.to_numeric(df["elapsedyrs"], errors="coerce").astype(int)

    # 3) fatalities
    df["ndam15"] = pd.to_numeric(df["ndam15"], errors="coerce").fillna(0).astype(int)
    df["log_deaths"] = np.log1p(df["ndam15"])  # dependent variable for OLS

    # 4) storm severity controls
    df["saffir_category"] = pd.to_numeric(df["masfem"], errors="coerce")
    df["max_wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["min_pressure"] = pd.to_numeric(df["min"], errors="coerce")

    # 5) property damage (skewed) -> log transform
    df["ind"] = pd.to_numeric(df["ind"], errors="coerce").fillna(0)
    df["log_property_damage"] = np.log1p(df["ind"])

    # 6) year (use 'alldeaths' as described in schema)
    df["year"] = pd.to_numeric(df["alldeaths"], errors="coerce")

    # After creating derived columns, drop rows that still have NA in any modeling column
    model_cols = [
        "name_mf_z",
        "female_name",
        "ndam15",
        "log_deaths",
        "saffir_category",
        "max_wind",
        "min_pressure",
        "log_property_damage",
        "year"
    ]
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit the primary statistical models to test whether more feminine hurricane names are associated
    with (a) higher fatalities (our proxy for fewer effective precautions) controlling for storm severity,
    and (b) as a robustness check, fit a count model appropriate for overdispersed casualty counts.

    Returns a dictionary with the fitted OLS and Negative Binomial model results objects.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure required columns exist
    required = [
        "log_deaths",
        "name_mf_z",
        "female_name",
        "saffir_category",
        "max_wind",
        "min_pressure",
        "log_property_damage",
        "ndam15",
        "year"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Build formula for OLS (dependent variable is log_deaths)
    formula = (
        "log_deaths ~ name_mf_z + female_name + saffir_category + max_wind + "
        "min_pressure + log_property_damage + year"
    )

    # Fit OLS with robust standard errors (HC3)
    ols_res = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Fit Negative Binomial on raw counts (ndam15) as a robustness check
    # Construct design matrix (explicitly add constant)
    exog_vars = [
        "name_mf_z",
        "female_name",
        "saffir_category",
        "max_wind",
        "min_pressure",
        "log_property_damage",
        "year"
    ]
    exog = sm.add_constant(df[exog_vars])
    endog = df["ndam15"].astype(int)

    # If all counts are zero or model fails to converge, handle exceptions
    try:
        nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
    except Exception as e:
        nb_res = None
        # Do not raise immediately; include error information in results

    # Return both fits; callers can inspect summary() on objects
    results = {
        "ols_robust": ols_res,
        "neg_binom": nb_res,
        "formula": formula
    }

    return results


