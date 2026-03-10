import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
csv_path = "teachingratings.csv"
df = pd.read_csv(csv_path)

# Basic cleaning
# Ensure categorical variables are treated as categories
cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype("category")

# Simple correlation
corr = df["beauty"].corr(df["eval"])

# Simple OLS
model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

# Multivariate OLS with controls
# Use students (participants) and avoid allstudents to reduce collinearity
formula = "eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(credits) + C(division) + students"
model_controls = smf.ols(formula, data=df).fit(cov_type="HC3")

# Cluster-robust SE by prof (instructor)
model_controls_cluster = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["prof"]})

# Extract stats
results = {
    "n": int(df.shape[0]),
    "corr_beauty_eval": corr,
    "simple_coef": float(model_simple.params["beauty"]),
    "simple_p": float(model_simple.pvalues["beauty"]),
    "controls_coef": float(model_controls.params["beauty"]),
    "controls_p": float(model_controls.pvalues["beauty"]),
    "controls_cluster_coef": float(model_controls_cluster.params["beauty"]),
    "controls_cluster_p": float(model_controls_cluster.pvalues["beauty"]),
}

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
