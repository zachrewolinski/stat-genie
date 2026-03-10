import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Basic cleaning
# Ensure categorical types
for col in ["feature3", "feature4", "feature7"]:
    df[col] = df[col].astype("category")

# Efficiency: nuts opened per second
# Avoid division by zero; feature6 min > 0 per metadata

df["efficiency"] = df["feature5"] / df["feature6"]

# Standardize naming for readability
# feature2: age; feature3: sex; feature7: help

# OLS model
model = smf.ols("efficiency ~ feature2 + C(feature3) + C(feature7)", data=df).fit()

# Alternative: count model with duration as offset (Poisson)
# Add a small constant to avoid log(0) for sessions with 0 nuts opened
# Use quasi-poisson via GLM with robust SE
import statsmodels.api as sm

df["nuts_opened"] = df["feature5"]
df["duration"] = df["feature6"]

glm_model = smf.glm(
    "nuts_opened ~ feature2 + C(feature3) + C(feature7)",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["duration"])  # model rate per second
).fit(cov_type="HC3")

# Summaries for reporting
result = {
    "n": int(df.shape[0]),
    "efficiency_mean": float(df["efficiency"].mean()),
    "efficiency_std": float(df["efficiency"].std()),
    "ols_params": model.params.to_dict(),
    "ols_pvalues": model.pvalues.to_dict(),
    "ols_r2": float(model.rsquared),
    "glm_params": glm_model.params.to_dict(),
    "glm_pvalues": glm_model.pvalues.to_dict(),
    "glm_deviance": float(glm_model.deviance),
}

with open("analysis_results.json", "w") as f:
    json.dump(result, f, indent=2)

print("OK")
