import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv("hurricane.csv")

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = [
    "masfem",
    "masfem_mturk",
    "gender_mf",
    "wind",
    "min",
    "category",
    "alldeaths",
    "ndam",
    "ndam15",
    "year",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Create log-transformed outcomes and controls due to skew

df["log_deaths"] = np.log1p(df["alldeaths"])
# Use log damage as control (normalized damage can be proxy for exposure/economic impact)

df["log_ndam"] = np.log1p(df["ndam"])

# Drop rows with missing key variables
base_cols = [
    "log_deaths",
    "masfem",
    "gender_mf",
    "wind",
    "min",
    "category",
    "log_ndam",
    "year",
]

base_df = df.dropna(subset=base_cols).copy()

# Model specifications
models = {}

# 1) Bivariate: log deaths ~ masfem
models["biv_masfem"] = smf.ols("log_deaths ~ masfem", data=base_df).fit()

# 2) Controls for storm intensity and damage
models["ctrl_masfem"] = smf.ols(
    "log_deaths ~ masfem + wind + min + category + log_ndam + year",
    data=base_df,
).fit()

# 3) Alternative masculinity/femininity measure (mturk)
models["ctrl_masfem_mturk"] = smf.ols(
    "log_deaths ~ masfem_mturk + wind + min + category + log_ndam + year",
    data=base_df.dropna(subset=["masfem_mturk"]).copy(),
).fit()

# 4) Binary gender indicator
models["ctrl_gender"] = smf.ols(
    "log_deaths ~ gender_mf + wind + min + category + log_ndam + year",
    data=base_df,
).fit()

# Collect key stats
summary = {}

for name, m in models.items():
    # Grab coefficient and p-value for focal variable
    if "masfem_mturk" in m.params.index:
        key = "masfem_mturk"
    elif "masfem" in m.params.index:
        key = "masfem"
    elif "gender_mf" in m.params.index:
        key = "gender_mf"
    else:
        key = None

    if key is None:
        continue
    summary[name] = {
        "n": int(m.nobs),
        "coef": float(m.params[key]),
        "se": float(m.bse[key]),
        "p": float(m.pvalues[key]),
        "r2": float(m.rsquared),
    }

# Also compute simple correlations with deaths (raw and log)
correlations = {}
for var in ["masfem", "masfem_mturk", "gender_mf"]:
    if var in df.columns:
        correlations[var] = {
            "corr_alldeaths": float(df[[var, "alldeaths"]].corr().iloc[0, 1]),
            "corr_log_deaths": float(df[[var, "log_deaths"]].corr().iloc[0, 1]),
        }

output = {
    "summary": summary,
    "correlations": correlations,
    "n_total": int(df.shape[0]),
}

print(json.dumps(output, indent=2))
