import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "hurricane.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure numeric columns are numeric
num_cols = [
    "masfem",
    "masfem_mturk",
    "alldeaths",
    "ndam",
    "ndam15",
    "min",
    "wind",
    "category",
    "year",
    "elapsedyrs",
    "gender_mf",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Create outcomes
# log(1 + deaths) to handle zeros and skew
# log(1 + ndam15) as damage proxy

df["log_deaths"] = np.log1p(df["alldeaths"].fillna(0))
df["log_ndam15"] = np.log1p(df["ndam15"].fillna(0))

# Define modeling dataset with key controls
model_cols = ["log_deaths", "masfem", "category", "wind", "min", "year"]
model_df = df[model_cols].dropna()

# Base model: log deaths ~ masfem
base_df = df[["log_deaths", "masfem"]].dropna()
base_model = smf.ols("log_deaths ~ masfem", data=base_df).fit(cov_type="HC3")

# Control model
control_model = smf.ols(
    "log_deaths ~ masfem + category + wind + min + year", data=model_df
).fit(cov_type="HC3")

# Alternative: use MTurk ratings
alt_cols = ["log_deaths", "masfem_mturk", "category", "wind", "min", "year"]
alt_df = df[alt_cols].dropna()
alt_model = smf.ols(
    "log_deaths ~ masfem_mturk + category + wind + min + year", data=alt_df
).fit(cov_type="HC3")

# Binary gender model
bin_cols = ["log_deaths", "gender_mf", "category", "wind", "min", "year"]
bin_df = df[bin_cols].dropna()
bin_model = smf.ols(
    "log_deaths ~ gender_mf + category + wind + min + year", data=bin_df
).fit(cov_type="HC3")

# Correlations
corr_pearson = df[["masfem", "log_deaths"]].corr(method="pearson").iloc[0, 1]
corr_spearman = df[["masfem", "log_deaths"]].corr(method="spearman").iloc[0, 1]

results = {
    "n_total": len(df),
    "n_base": int(base_df.shape[0]),
    "n_control": int(model_df.shape[0]),
    "n_alt": int(alt_df.shape[0]),
    "n_bin": int(bin_df.shape[0]),
    "base_coef": base_model.params.get("masfem"),
    "base_p": base_model.pvalues.get("masfem"),
    "control_coef": control_model.params.get("masfem"),
    "control_p": control_model.pvalues.get("masfem"),
    "alt_coef": alt_model.params.get("masfem_mturk"),
    "alt_p": alt_model.pvalues.get("masfem_mturk"),
    "bin_coef": bin_model.params.get("gender_mf"),
    "bin_p": bin_model.pvalues.get("gender_mf"),
    "corr_pearson": corr_pearson,
    "corr_spearman": corr_spearman,
}

print(json.dumps(results, indent=2))
