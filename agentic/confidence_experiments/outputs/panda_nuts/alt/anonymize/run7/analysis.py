import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"

df = pd.read_csv(csv_path)

# Rename columns for clarity
rename_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer_type",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help",
}

df = df.rename(columns=rename_map)

# Basic cleaning: ensure numeric types
for col in ["age", "nuts_opened", "duration_sec"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Efficiency: nuts per second
# Avoid division by zero; duration has min 2.5 in metadata

df["efficiency"] = df["nuts_opened"] / df["duration_sec"]

# Drop rows with missing key fields
analysis_df = df.dropna(subset=["efficiency", "age", "sex", "help"]).copy()

# Fit linear model with categorical predictors
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=analysis_df).fit(cov_type="HC3")

# Sensitivity: log efficiency (add small constant to avoid log(0))
analysis_df["log_efficiency"] = np.log(analysis_df["efficiency"] + 1e-6)
model_log = smf.ols("log_efficiency ~ age + C(sex) + C(help)", data=analysis_df).fit(cov_type="HC3")

# Collect key stats
results = {
    "n": int(analysis_df.shape[0]),
    "r2": float(model.rsquared),
    "adj_r2": float(model.rsquared_adj),
    "params": model.params.to_dict(),
    "pvalues": model.pvalues.to_dict(),
    "log_pvalues": model_log.pvalues.to_dict(),
    "log_params": model_log.params.to_dict(),
}

# Write to json for inspection
with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Rows:", analysis_df.shape[0])
print(model.summary())
print("\nLog model summary:\n")
print(model_log.summary())
