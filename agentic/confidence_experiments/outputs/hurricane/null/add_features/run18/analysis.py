import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv("hurricane.csv")

# Core transformations
# Use log1p to handle zeros in deaths and damages
for col in ["alldeaths", "ndam15"]:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col])

# Build modeling frame with relevant columns
cols = [
    "log_alldeaths",
    "masfem",
    "gender_mf",
    "wind",
    "min",
    "category",
    "log_ndam15",
]
# Keep only columns that exist
cols = [c for c in cols if c in df.columns]
model_df = df[cols].dropna().copy()

# Helper to extract coef and p

def coef_p(model, term):
    if term in model.params.index:
        return float(model.params[term]), float(model.pvalues[term])
    return None, None

results = {}

if "log_alldeaths" in model_df.columns and "masfem" in model_df.columns:
    # Model with damage control
    formula1 = "log_alldeaths ~ masfem + wind + min + category"
    if "log_ndam15" in model_df.columns:
        formula1 += " + log_ndam15"
    m1 = smf.ols(formula1, data=model_df).fit(cov_type="HC3")
    results["m1"] = {
        "formula": formula1,
        "n": int(m1.nobs),
        "masfem_coef": coef_p(m1, "masfem")[0],
        "masfem_p": coef_p(m1, "masfem")[1],
        "r2": float(m1.rsquared),
    }

    # Interaction with wind (severity)
    formula2 = "log_alldeaths ~ masfem * wind + min + category"
    if "log_ndam15" in model_df.columns:
        formula2 += " + log_ndam15"
    m2 = smf.ols(formula2, data=model_df).fit(cov_type="HC3")
    results["m2"] = {
        "formula": formula2,
        "n": int(m2.nobs),
        "masfem_coef": coef_p(m2, "masfem")[0],
        "masfem_p": coef_p(m2, "masfem")[1],
        "interaction_coef": coef_p(m2, "masfem:wind")[0],
        "interaction_p": coef_p(m2, "masfem:wind")[1],
        "r2": float(m2.rsquared),
    }

    # Model without damage control
    formula3 = "log_alldeaths ~ masfem + wind + min + category"
    m3 = smf.ols(formula3, data=model_df).fit(cov_type="HC3")
    results["m3"] = {
        "formula": formula3,
        "n": int(m3.nobs),
        "masfem_coef": coef_p(m3, "masfem")[0],
        "masfem_p": coef_p(m3, "masfem")[1],
        "r2": float(m3.rsquared),
    }

    formula4 = "log_alldeaths ~ masfem * wind + min + category"
    m4 = smf.ols(formula4, data=model_df).fit(cov_type="HC3")
    results["m4"] = {
        "formula": formula4,
        "n": int(m4.nobs),
        "masfem_coef": coef_p(m4, "masfem")[0],
        "masfem_p": coef_p(m4, "masfem")[1],
        "interaction_coef": coef_p(m4, "masfem:wind")[0],
        "interaction_p": coef_p(m4, "masfem:wind")[1],
        "r2": float(m4.rsquared),
    }

    # Binary gender indicator model
    if "gender_mf" in model_df.columns:
        formula5 = "log_alldeaths ~ gender_mf + wind + min + category"
        if "log_ndam15" in model_df.columns:
            formula5 += " + log_ndam15"
        m5 = smf.ols(formula5, data=model_df).fit(cov_type="HC3")
        results["m5"] = {
            "formula": formula5,
            "n": int(m5.nobs),
            "gender_mf_coef": coef_p(m5, "gender_mf")[0],
            "gender_mf_p": coef_p(m5, "gender_mf")[1],
            "r2": float(m5.rsquared),
        }

    # Simple correlation
    results["corr"] = float(model_df["masfem"].corr(model_df["log_alldeaths"]))

print(json.dumps(results, indent=2))
