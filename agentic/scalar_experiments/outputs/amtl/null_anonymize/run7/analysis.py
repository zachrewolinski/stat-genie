import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Rename for clarity
_df = _df.rename(columns={
    "feature1": "tooth_class",
    "feature3": "missing",
    "feature4": "observable",
    "feature5": "age",
    "feature7": "sex",
    "feature8": "genus",
})

# Basic cleaning
_df = _df.dropna(subset=["tooth_class", "missing", "observable", "age", "sex", "genus"])
_df = _df[_df["observable"] > 0]
_df = _df[_df["missing"] >= 0]
_df = _df[_df["observable"] >= _df["missing"]]

_df["present"] = _df["observable"] - _df["missing"]
_df["is_human"] = (_df["genus"] == "Homo sapiens").astype(int)

# Fit binomial GLM: missing vs present with covariates
formula = "missing + present ~ is_human + age + sex + C(tooth_class)"
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial())
result = model.fit()

coef = result.params.get("is_human", float("nan"))
pval = result.pvalues.get("is_human", float("nan"))

# Determine response and scale
if math.isnan(coef) or math.isnan(pval):
    response = "No"
    scale = 0
else:
    if coef > 0 and pval < 0.05:
        response = "Yes"
        strength = min(1.0, (-math.log10(max(pval, 1e-300))) / 3.0)
        scale = 50 + round(50 * strength)
    else:
        response = "No"
        # Stronger "No" for larger p-values and/or negative effects
        if pval >= 0.05:
            scale = round(50 * (0.05 / pval))
        else:
            strength = min(1.0, (-math.log10(max(pval, 1e-300))) / 3.0)
            scale = 50 - round(50 * strength)

scale = max(0, min(100, int(scale)))

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump({"response": response, "scale": scale}, f)
