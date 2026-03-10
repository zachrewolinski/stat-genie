import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
DATA_PATH = "crofoot.csv"
df = pd.read_csv(DATA_PATH)

# Create predictors
# Relative group size: focal - other
# Contest location advantage: positive if contest is closer to focal home range center
# (since smaller distance means closer)
df["size_diff"] = df["n_focal"] - df["n_other"]
df["loc_adv"] = df["dist_other"] - df["dist_focal"]

# Standardize predictors for comparable coefficients (optional)
# But keep raw for interpretability; we'll also compute standardized for effect sizes

# Logistic regression
X = df[["size_diff", "loc_adv"]]
X = sm.add_constant(X)
y = df["win"]

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also fit model with standardized predictors
X_std = df[["size_diff", "loc_adv"]].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
X_std = sm.add_constant(X_std)
model_std = sm.Logit(y, X_std)
result_std = model_std.fit(disp=False)

# Compute pseudo-R2 (McFadden)
ll_null = sm.Logit(y, sm.add_constant(pd.DataFrame({"intercept": np.ones(len(df))}))).fit(disp=False).llf
ll_model = result.llf
pseudo_r2 = 1 - ll_model / ll_null

# Simple comparisons: win rate by location advantage sign and size_diff sign
loc_pos = df[df["loc_adv"] > 0]["win"].mean()
loc_neg = df[df["loc_adv"] <= 0]["win"].mean()
size_pos = df[df["size_diff"] > 0]["win"].mean()
size_neg = df[df["size_diff"] <= 0]["win"].mean()

# Output summary
print("N:", len(df))
print("Win rate overall:", df["win"].mean())
print("Logit coefficients (raw):")
print(result.summary())
print("Logit coefficients (standardized):")
print(result_std.summary())
print("McFadden pseudo-R2:", pseudo_r2)
print("Win rate loc_adv>0 vs <=0:", loc_pos, loc_neg)
print("Win rate size_diff>0 vs <=0:", size_pos, size_neg)

# Save key results to a JSON-like dict (for easy parsing if needed)
import json
out = {
    "n": int(len(df)),
    "win_rate": float(df["win"].mean()),
    "coef": {k: float(v) for k, v in result.params.items()},
    "pvalues": {k: float(v) for k, v in result.pvalues.items()},
    "coef_std": {k: float(v) for k, v in result_std.params.items()},
    "pvalues_std": {k: float(v) for k, v in result_std.pvalues.items()},
    "pseudo_r2": float(pseudo_r2),
    "win_rate_loc_adv_pos": float(loc_pos),
    "win_rate_loc_adv_nonpos": float(loc_neg),
    "win_rate_size_diff_pos": float(size_pos),
    "win_rate_size_diff_nonpos": float(size_neg),
}
with open("analysis_results.json", "w") as f:
    json.dump(out, f, indent=2)
