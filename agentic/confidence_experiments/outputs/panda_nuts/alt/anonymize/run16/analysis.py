import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Rename for clarity
cols = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help",
}
df = df.rename(columns=cols)

# Basic cleaning
# Ensure categorical
for c in ["sex", "hammer", "help"]:
    df[c] = df[c].astype("category")

# Efficiency: nuts opened per second (rate). Avoid division by zero.
df = df[df["duration_sec"] > 0].copy()
df["efficiency"] = df["nuts_opened"] / df["duration_sec"]

# Also compute per minute for interpretability
per_minute = df["efficiency"] * 60.0

# Summary stats
summary = {
    "n_rows": int(len(df)),
    "efficiency_mean_per_sec": float(df["efficiency"].mean()),
    "efficiency_std_per_sec": float(df["efficiency"].std()),
    "efficiency_median_per_sec": float(df["efficiency"].median()),
    "efficiency_mean_per_min": float(per_minute.mean()),
    "age_min": float(df["age"].min()),
    "age_max": float(df["age"].max()),
    "sex_counts": df["sex"].value_counts().to_dict(),
    "help_counts": df["help"].value_counts().to_dict(),
}

# OLS model with categorical sex and help. Use robust SEs.
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Also check model with log efficiency to reduce skew (add small constant)
df["log_efficiency"] = np.log(df["efficiency"] + 1e-6)
log_model = smf.ols("log_efficiency ~ age + C(sex) + C(help)", data=df).fit(cov_type="HC3")

# Extract key results
results = {
    "ols_params": model.params.to_dict(),
    "ols_pvalues": model.pvalues.to_dict(),
    "ols_r2": float(model.rsquared),
    "ols_adj_r2": float(model.rsquared_adj),
    "log_params": log_model.params.to_dict(),
    "log_pvalues": log_model.pvalues.to_dict(),
    "log_r2": float(log_model.rsquared),
    "log_adj_r2": float(log_model.rsquared_adj),
}

# Group means for context
group_means = {
    "efficiency_mean_by_sex": df.groupby("sex")["efficiency"].mean().to_dict(),
    "efficiency_mean_by_help": df.groupby("help")["efficiency"].mean().to_dict(),
}

# Save results
output = {
    "summary": summary,
    "results": results,
    "group_means": group_means,
}

with open("analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

# Print concise output for review
print(json.dumps(output, indent=2))
