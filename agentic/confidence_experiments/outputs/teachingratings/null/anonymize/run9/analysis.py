import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "teachingratings.csv"
df = pd.read_csv(path)

# Basic cleanup: ensure categorical types
cat_cols = ["feature2", "feature4", "feature5", "feature8", "feature9", "feature10"]
for c in cat_cols:
    df[c] = df[c].astype("category")

# Core variables
beauty = df["feature6"]
rating = df["feature7"]

# Pearson correlation
corr = beauty.corr(rating)

# Simple OLS
model_simple = smf.ols("feature7 ~ feature6", data=df).fit()

# Adjusted OLS with controls
# Use categorical variables via C(), keep numeric as-is
formula = (
    "feature7 ~ feature6 + feature3 + C(feature4) + C(feature2) + C(feature5) "
    "+ C(feature8) + C(feature9) + C(feature10) + feature11 + feature12"
)
model_adj = smf.ols(formula, data=df).fit()

# Extract key stats
simple_coef = model_simple.params["feature6"]
simple_p = model_simple.pvalues["feature6"]
simple_ci = model_simple.conf_int().loc["feature6"].tolist()

adj_coef = model_adj.params["feature6"]
adj_p = model_adj.pvalues["feature6"]
adj_ci = model_adj.conf_int().loc["feature6"].tolist()

# Effect size: change in rating per 1 SD in beauty
beauty_sd = beauty.std()
rate_sd = rating.std()
std_effect = simple_coef * beauty_sd

# Save analysis summary to JSON for later reading
summary = {
    "n": int(len(df)),
    "corr": float(corr),
    "simple_coef": float(simple_coef),
    "simple_p": float(simple_p),
    "simple_ci": [float(simple_ci[0]), float(simple_ci[1])],
    "adj_coef": float(adj_coef),
    "adj_p": float(adj_p),
    "adj_ci": [float(adj_ci[0]), float(adj_ci[1])],
    "beauty_sd": float(beauty_sd),
    "rating_sd": float(rate_sd),
    "std_effect": float(std_effect),
    "simple_r2": float(model_simple.rsquared),
    "adj_r2": float(model_adj.rsquared),
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
