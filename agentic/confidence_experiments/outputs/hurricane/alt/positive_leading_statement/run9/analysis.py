import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("hurricane.csv")

# Basic cleaning
# Ensure numeric
for col in ["masfem", "masfem_mturk", "wind", "min", "ndam15", "ndam", "alldeaths", "year", "category"]:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors="coerce")

# Create transformed variables
_df["log_deaths"] = np.log1p(_df["alldeaths"])
_df["log_ndam15"] = np.log1p(_df["ndam15"])

# Drop rows with missing key fields
key_cols = ["log_deaths", "masfem", "wind", "min", "log_ndam15", "year", "category"]
_df = _df.dropna(subset=key_cols).copy()

# Model 1: simple bivariate
m1 = smf.ols("log_deaths ~ masfem", data=_df).fit(cov_type="HC3")

# Model 2: controls for intensity and year
m2 = smf.ols("log_deaths ~ masfem + wind + min + log_ndam15 + year", data=_df).fit(cov_type="HC3")

# Model 3: include category instead of wind/min to check robustness
m3 = smf.ols("log_deaths ~ masfem + category + log_ndam15 + year", data=_df).fit(cov_type="HC3")

# Binary gender indicator model
m4 = smf.ols("log_deaths ~ gender_mf + wind + min + log_ndam15 + year", data=_df).fit(cov_type="HC3")

# Correlation
corr = _df[["masfem", "alldeaths"]].corr().iloc[0,1]

# Effect size: 1 SD increase in masfem from model 2
masfem_sd = _df["masfem"].std()
coef_m2 = m2.params.get("masfem", np.nan)

# Convert log effect to percent change in deaths for 1 SD increase
pct_change_1sd = (np.exp(coef_m2 * masfem_sd) - 1) * 100

# Collect results
results = {
    "n": int(_df.shape[0]),
    "corr_masfem_deaths": float(corr),
    "m1_coef": float(m1.params["masfem"]),
    "m1_p": float(m1.pvalues["masfem"]),
    "m2_coef": float(m2.params["masfem"]),
    "m2_p": float(m2.pvalues["masfem"]),
    "m3_coef": float(m3.params["masfem"]),
    "m3_p": float(m3.pvalues["masfem"]),
    "m4_coef": float(m4.params["gender_mf"]),
    "m4_p": float(m4.pvalues["gender_mf"]),
    "pct_change_deaths_per_1sd_masfem_m2": float(pct_change_1sd),
}

print(json.dumps(results, indent=2))
