import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Rename columns to meaningful names
col_map = {
    "feature1": "id",
    "feature2": "age",
    "feature3": "sex",
    "feature4": "hammer",
    "feature5": "nuts_opened",
    "feature6": "duration_sec",
    "feature7": "help",
}

df = df.rename(columns=col_map)

# Efficiency: nuts per second
# Guard against division by zero (not expected from metadata)
df["efficiency"] = df["nuts_opened"] / df["duration_sec"]

# Basic summaries
summary = {
    "n_rows": len(df),
    "efficiency_summary": df["efficiency"].describe().to_dict(),
    "age_summary": df["age"].describe().to_dict(),
    "sex_counts": df["sex"].value_counts().to_dict(),
    "help_counts": df["help"].value_counts().to_dict(),
}

# Ensure categorical types
for col in ["sex", "help"]:
    df[col] = df[col].astype("category")

# Regression: efficiency ~ age + sex + help
# Use OLS; also compute HC3 robust SE as sensitivity
model = smf.ols("efficiency ~ age + C(sex) + C(help)", data=df).fit()
model_hc3 = model.get_robustcov_results(cov_type="HC3")

# Also check log efficiency to reduce skew (add small constant)
# Small constant based on min positive
min_eff = df["efficiency"].min()
const = min_eff * 0.5 if min_eff > 0 else 1e-6

df["log_efficiency"] = np.log(df["efficiency"] + const)
log_model = smf.ols("log_efficiency ~ age + C(sex) + C(help)", data=df).fit()
log_model_hc3 = log_model.get_robustcov_results(cov_type="HC3")

# Package results
results = {
    "summary": summary,
    "ols": {
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "nobs": int(model.nobs),
    },
    "ols_hc3": {
        "params": model_hc3.params.tolist(),
        "pvalues": model_hc3.pvalues.tolist(),
    },
    "log_ols": {
        "params": log_model.params.to_dict(),
        "pvalues": log_model.pvalues.to_dict(),
        "r2": log_model.rsquared,
        "adj_r2": log_model.rsquared_adj,
        "nobs": int(log_model.nobs),
    },
    "log_ols_hc3": {
        "params": log_model_hc3.params.tolist(),
        "pvalues": log_model_hc3.pvalues.tolist(),
    },
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Wrote analysis_results.json")
