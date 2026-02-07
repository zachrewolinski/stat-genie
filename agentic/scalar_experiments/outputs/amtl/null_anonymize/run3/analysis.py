import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv("amtl.csv")

# Rename for clarity
_df = _df.rename(
    columns={
        "feature1": "tooth_class",
        "feature3": "missing",
        "feature4": "total",
        "feature5": "age",
        "feature7": "sex",
        "feature8": "genus",
    }
)

# Basic cleaning
_df = _df.dropna(subset=["tooth_class", "missing", "total", "age", "sex", "genus"]).copy()
_df = _df[_df["total"] > 0]

# Human indicator
_df["human"] = _df["genus"].astype(str).str.contains("Homo", case=False, regex=False).astype(int)

# Proportion outcome with binomial weights
_df["prop"] = _df["missing"] / _df["total"]

# Fit binomial GLM with weights
model = smf.glm(
    "prop ~ human + age + sex + C(tooth_class)",
    data=_df,
    family=sm.families.Binomial(),
    var_weights=_df["total"],
).fit()

coef = model.params.get("human", np.nan)
se = model.bse.get("human", np.nan)
if np.isfinite(coef) and np.isfinite(se) and se > 0:
    z = coef / se
else:
    z = np.nan

# Map z-score to Likert scale
if not np.isfinite(z):
    score = 0
else:
    strength = abs(z)
    if strength >= 5:
        base = 100
    elif strength >= 4:
        base = 90
    elif strength >= 3:
        base = 75
    elif strength >= 2:
        base = 60
    elif strength >= 1.5:
        base = 45
    elif strength >= 1:
        base = 30
    elif strength >= 0.5:
        base = 15
    else:
        base = 5
    score = base if coef > 0 else -base

# Write conclusion
with open("conclusion.txt", "w", encoding="utf-8") as f:
    f.write(str(int(score)))

# Optional: print key results for logging
print(model.summary())
print("human coef", coef, "z", z, "score", score)
