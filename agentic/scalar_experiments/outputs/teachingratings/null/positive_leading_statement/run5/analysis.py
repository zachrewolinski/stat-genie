import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = "teachingratings.csv"

df = pd.read_csv(DATA_PATH)

# Basic cleaning
# Ensure expected columns exist
expected_cols = {
    "beauty", "eval", "age", "gender", "minority", "native", "tenure",
    "division", "credits", "students", "allstudents"
}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Model 1: simple regression eval ~ beauty
X1 = sm.add_constant(df[["beauty"]])
model1 = sm.OLS(df["eval"], X1).fit(cov_type="HC3")

# Model 2: with controls (categorical dummies)
cat_cols = ["gender", "minority", "native", "tenure", "division", "credits"]
num_cols = ["beauty", "age", "students", "allstudents"]

X2 = df[num_cols + cat_cols].copy()
X2 = pd.get_dummies(X2, columns=cat_cols, drop_first=True)
X2 = sm.add_constant(X2)
model2 = sm.OLS(df["eval"], X2).fit(cov_type="HC3")

# Extract results for beauty
res = {
    "n": int(df.shape[0]),
    "simple": {
        "coef": float(model1.params["beauty"]),
        "pval": float(model1.pvalues["beauty"]),
        "ci_low": float(model1.conf_int().loc["beauty", 0]),
        "ci_high": float(model1.conf_int().loc["beauty", 1]),
        "r2": float(model1.rsquared),
    },
    "controls": {
        "coef": float(model2.params.get("beauty", np.nan)),
        "pval": float(model2.pvalues.get("beauty", np.nan)),
        "ci_low": float(model2.conf_int().loc["beauty", 0]),
        "ci_high": float(model2.conf_int().loc["beauty", 1]),
        "r2": float(model2.rsquared),
    },
}

# Simple correlation as supporting evidence
res["corr"] = float(df["beauty"].corr(df["eval"]))

print(json.dumps(res, indent=2))
