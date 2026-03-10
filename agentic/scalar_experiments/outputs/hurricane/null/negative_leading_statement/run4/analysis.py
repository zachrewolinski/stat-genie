import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Create variables
# Use log1p for deaths to handle zeros and skew
if "alldeaths" not in df.columns:
    raise SystemExit("alldeaths column missing")

df["log_deaths"] = np.log1p(df["alldeaths"])

# Define helper to fit OLS with robust SEs

def fit_ols(data, y, X_cols):
    X = sm.add_constant(data[X_cols], has_constant="add")
    model = sm.OLS(data[y], X, missing="drop")
    res = model.fit(cov_type="HC3")
    return res

# Define helper to fit Poisson GLM with robust SEs

def fit_poisson(data, y, X_cols):
    X = sm.add_constant(data[X_cols], has_constant="add")
    model = sm.GLM(data[y], X, family=sm.families.Poisson(), missing="drop")
    res = model.fit(cov_type="HC3")
    return res

# Columns for controls
control_cols = ["category", "wind", "min", "year"]

# Prepare datasets for models
cols_needed_m1 = ["masfem", "log_deaths"]
cols_needed_m2 = ["masfem", "log_deaths"] + control_cols
cols_needed_p = ["masfem", "alldeaths"] + control_cols

m1 = fit_ols(df.dropna(subset=cols_needed_m1), "log_deaths", ["masfem"])

m2_data = df.dropna(subset=cols_needed_m2)
m2 = fit_ols(m2_data, "log_deaths", ["masfem"] + control_cols)

p_data = df.dropna(subset=cols_needed_p)
poisson = fit_poisson(p_data, "alldeaths", ["masfem"] + control_cols)

# Simple correlations
corr_raw = df["masfem"].corr(df["alldeaths"], method="pearson")
corr_log = df["masfem"].corr(df["log_deaths"], method="pearson")

results = {
    "n_total": int(len(df)),
    "n_m1": int(m1.nobs),
    "n_m2": int(m2.nobs),
    "n_poisson": int(poisson.nobs),
    "corr_masfem_alldeaths": float(corr_raw),
    "corr_masfem_logdeaths": float(corr_log),
    "ols_m1": {
        "coef": float(m1.params["masfem"]),
        "pvalue": float(m1.pvalues["masfem"]),
        "ci_low": float(m1.conf_int().loc["masfem", 0]),
        "ci_high": float(m1.conf_int().loc["masfem", 1]),
    },
    "ols_m2": {
        "coef": float(m2.params["masfem"]),
        "pvalue": float(m2.pvalues["masfem"]),
        "ci_low": float(m2.conf_int().loc["masfem", 0]),
        "ci_high": float(m2.conf_int().loc["masfem", 1]),
    },
    "poisson": {
        "coef": float(poisson.params["masfem"]),
        "pvalue": float(poisson.pvalues["masfem"]),
        "ci_low": float(poisson.conf_int().loc["masfem", 0]),
        "ci_high": float(poisson.conf_int().loc["masfem", 1]),
    },
}

print(json.dumps(results, indent=2))
