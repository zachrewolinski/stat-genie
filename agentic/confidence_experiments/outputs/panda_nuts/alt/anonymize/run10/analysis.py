import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "panda_nuts.csv"
df = pd.read_csv(csv_path)

# Rename for clarity
# feature1: individual ID
# feature2: age
# feature3: sex
# feature4: hammer type
# feature5: nuts opened
# feature6: session duration (s)
# feature7: received help (y/N)

# Compute efficiency: nuts opened per second
# Guard against zero/negative duration just in case (shouldn't happen)
df = df.copy()
df["efficiency"] = df["feature5"] / df["feature6"]

# Basic summaries
summary = {
    "n_rows": len(df),
    "n_individuals": df["feature1"].nunique(),
    "efficiency_mean": df["efficiency"].mean(),
    "efficiency_median": df["efficiency"].median(),
    "efficiency_zero_share": float((df["efficiency"] == 0).mean()),
}

# Encode help as categorical
# feature7 is 'y' or 'N'

# Model 1: OLS with cluster-robust SE by individual
model = smf.ols("efficiency ~ feature2 + C(feature3) + C(feature7)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["feature1"]}
)

# Model 2 (sensitivity): log1p efficiency to reduce skew
# Add small offset inside log1p, but efficiency can be zero so log1p is fine
model_log = smf.ols("np.log1p(efficiency) ~ feature2 + C(feature3) + C(feature7)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["feature1"]}
)

# Extract key results
results = {
    "summary": summary,
    "model_params": model.params.to_dict(),
    "model_pvalues": model.pvalues.to_dict(),
    "model_log_params": model_log.params.to_dict(),
    "model_log_pvalues": model_log.pvalues.to_dict(),
}

print(json.dumps(results, indent=2, sort_keys=True))
