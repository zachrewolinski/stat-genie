import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Outcome: focal group win indicator
win = df["m_focal"].astype(int)

# Use metadata-based mapping from info.json descriptions
size_focal = df["f_other"].astype(float)
size_other = df["win"].astype(float)

# Distances from each group's home range center
# (metadata indicates these columns hold distances for focal and other groups)
dist_focal = df["m_other"].astype(float)
dist_other = df["n_focal"].astype(float)

# Relative group size: focal minus other
rel_size = size_focal - size_other

# Location advantage: positive when focal is closer to its home range center
loc_adv = dist_other - dist_focal

# Build logistic regression model
X = pd.DataFrame({"rel_size": rel_size, "loc_adv": loc_adv})
X = sm.add_constant(X)

model = sm.Logit(win, X, missing="drop")
result = model.fit(disp=False)

# Compute odds ratios and 95% CI
params = result.params
conf = result.conf_int()
odds_ratios = np.exp(params)
conf_or = np.exp(conf)

print("N:", len(df))
print("Win rate:", win.mean())
print("Logit coefficients:")
print(result.params)
print("p-values:")
print(result.pvalues)
print("Odds ratios:")
print(odds_ratios)
print("OR 95% CI:")
print(conf_or)

# Simple descriptive checks
# group win rate by relative size sign
rel_size_sign = pd.cut(rel_size, bins=[-np.inf, -0.5, 0.5, np.inf], labels=["smaller", "similar", "larger"])
summary = df.copy()
summary["rel_size"] = rel_size
summary["loc_adv"] = loc_adv
summary["rel_size_group"] = rel_size_sign
summary["win"] = win
print("\nWin rate by relative size group:")
print(summary.groupby("rel_size_group")["win"].mean())

# Win rate by location advantage terciles
loc_tercile = pd.qcut(loc_adv, 3, labels=["low", "mid", "high"])
summary["loc_adv_group"] = loc_tercile
print("\nWin rate by location advantage tercile:")
print(summary.groupby("loc_adv_group")["win"].mean())

# Save key results for easier reading
output = {
    "coef": result.params.to_dict(),
    "pvalues": result.pvalues.to_dict(),
    "odds_ratios": odds_ratios.to_dict(),
    "or_ci_lower": conf_or[0].to_dict(),
    "or_ci_upper": conf_or[1].to_dict(),
    "win_rate": win.mean(),
}

import json
with open("analysis_output.json", "w") as f:
    json.dump(output, f, indent=2)
