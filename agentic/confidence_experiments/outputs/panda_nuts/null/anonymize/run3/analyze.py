import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "panda_nuts.csv"
df = pd.read_csv(path)

# Map columns
# feature2: age, feature3: sex, feature7: help, feature5: nuts opened, feature6: duration (sec)

# Basic cleaning
# Ensure numeric types
for col in ["feature2", "feature5", "feature6"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing essentials or nonpositive duration
clean = df.dropna(subset=["feature2", "feature3", "feature7", "feature5", "feature6"]).copy()
clean = clean[clean["feature6"] > 0].copy()

# Define efficiency (nuts per second)
clean["efficiency"] = clean["feature5"] / clean["feature6"]

# Encode categorical predictors
# sex: m/f; help: y/N (note capital N)
clean["sex"] = clean["feature3"].astype("category")
clean["help"] = clean["feature7"].astype("category")

# Build model: efficiency ~ age + sex + help
# Use robust (HC3) SEs due to heteroskedasticity concerns
model = smf.ols("efficiency ~ feature2 + C(sex) + C(help)", data=clean).fit(cov_type="HC3")

# Also check log-efficiency to reduce skew (add small constant)
clean["log_eff"] = np.log1p(clean["efficiency"])
log_model = smf.ols("log_eff ~ feature2 + C(sex) + C(help)", data=clean).fit(cov_type="HC3")

# Spearman correlation for age vs efficiency
age_eff_spearman = clean[["feature2", "efficiency"]].corr(method="spearman").iloc[0, 1]

results = {
    "n_rows": int(len(clean)),
    "efficiency_mean": float(clean["efficiency"].mean()),
    "efficiency_median": float(clean["efficiency"].median()),
    "model_params": model.params.to_dict(),
    "model_pvalues": model.pvalues.to_dict(),
    "log_model_params": log_model.params.to_dict(),
    "log_model_pvalues": log_model.pvalues.to_dict(),
    "age_eff_spearman": float(age_eff_spearman),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
