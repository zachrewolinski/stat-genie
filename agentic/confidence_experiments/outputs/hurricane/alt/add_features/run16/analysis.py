import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Derived variables
# Use log1p to handle zero deaths and skewness
if "alldeaths" in df.columns:
    df["log_deaths"] = np.log1p(df["alldeaths"])

# Use inflation-adjusted damage if present
if "ndam15" in df.columns:
    df["log_dam15"] = np.log1p(df["ndam15"])

# Basic checks
print("rows", len(df))
print("deaths summary", df["alldeaths"].describe())

# Correlation between femininity and deaths
corr = df[["masfem", "alldeaths"]].corr().iloc[0, 1]
print("corr_masfem_alldeaths", corr)

# Simple model: log deaths ~ femininity
m1 = smf.ols("log_deaths ~ masfem", data=df).fit(cov_type="HC3")
print("m1_coef", m1.params.get("masfem"), "m1_p", m1.pvalues.get("masfem"))

# Main model: control for storm intensity and year
formula = "log_deaths ~ masfem + category + wind + min + log_dam15 + year"
m2 = smf.ols(formula, data=df).fit(cov_type="HC3")
print("m2_coef", m2.params.get("masfem"), "m2_p", m2.pvalues.get("masfem"))

# Binary gender model for robustness
m3 = smf.ols("log_deaths ~ gender_mf + category + wind + min + log_dam15 + year", data=df).fit(cov_type="HC3")
print("m3_coef", m3.params.get("gender_mf"), "m3_p", m3.pvalues.get("gender_mf"))

# Additional model without damage (since damage could be outcome-related)
formula2 = "log_deaths ~ masfem + category + wind + min + year"
m4 = smf.ols(formula2, data=df).fit(cov_type="HC3")
print("m4_coef", m4.params.get("masfem"), "m4_p", m4.pvalues.get("masfem"))

# Save key results to a small dict for later use (optional)
results = {
    "corr_masfem_alldeaths": float(corr),
    "m1_coef": float(m1.params.get("masfem")),
    "m1_p": float(m1.pvalues.get("masfem")),
    "m2_coef": float(m2.params.get("masfem")),
    "m2_p": float(m2.pvalues.get("masfem")),
    "m3_coef": float(m3.params.get("gender_mf")),
    "m3_p": float(m3.pvalues.get("gender_mf")),
    "m4_coef": float(m4.params.get("masfem")),
    "m4_p": float(m4.pvalues.get("masfem")),
}

import json
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

