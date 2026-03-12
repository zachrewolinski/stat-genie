import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv("hurricane.csv")

# Basic cleaning
for col in ["alldeaths", "ndam15", "ndam"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Derived variables
# Use +1 to handle zero deaths
if "alldeaths" in df.columns:
    df["log_deaths"] = np.log(df["alldeaths"] + 1)
if "ndam15" in df.columns:
    df["log_dam15"] = np.log(df["ndam15"] + 1)

# Variables of interest
masfem = "masfem"
masfem_alt = "masfem_mturk"

# Severity controls (common in literature)
controls = [c for c in ["category", "wind", "min", "year"] if c in df.columns]

results = {}

# Simple correlations
for target in ["alldeaths", "log_deaths", "ndam15", "log_dam15"]:
    if target in df.columns:
        for mf in [masfem, masfem_alt, "gender_mf"]:
            if mf in df.columns:
                corr = df[[target, mf]].corr().iloc[0, 1]
                results.setdefault("correlations", []).append({
                    "target": target,
                    "predictor": mf,
                    "corr": corr,
                })

# Regression helper

def run_ols(y, xvars):
    X = df[xvars].copy()
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(df[y], X, missing="drop").fit(cov_type="HC3")
    return {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "conf_int": model.conf_int().to_dict(),
    }

# Base models without interactions
for y in ["log_deaths", "log_dam15"]:
    if y in df.columns:
        for mf in [masfem, masfem_alt, "gender_mf"]:
            if mf in df.columns:
                xvars = [mf] + controls
                results.setdefault("models", []).append({
                    "y": y,
                    "x": xvars,
                    "result": run_ols(y, xvars),
                })

# Interaction with severity: masfem * category (if category exists)
if "category" in df.columns:
    for y in ["log_deaths", "log_dam15"]:
        if y in df.columns and masfem in df.columns:
            df["masfem_x_category"] = df[masfem] * df["category"]
            xvars = [masfem, "category", "masfem_x_category"] + [c for c in controls if c not in ["category"]]
            results.setdefault("models_interaction", []).append({
                "y": y,
                "x": xvars,
                "result": run_ols(y, xvars),
            })

# Save results to json for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Wrote analysis_results.json with", len(results.get("models", [])), "models")
