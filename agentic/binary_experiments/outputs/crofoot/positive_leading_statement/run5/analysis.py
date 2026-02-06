import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = "crofoot.csv"
df = pd.read_csv(path)

# Create predictors: relative group size and relative location (focal closer to its center)
df["size_diff"] = df["n_focal"] - df["n_other"]
# Positive means focal is closer to its home range center than the other group is to theirs
# (lower distance = closer)
df["loc_adv"] = df["dist_other"] - df["dist_focal"]

# Standardize predictors for comparability
for col in ["size_diff", "loc_adv"]:
    df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# Logistic regression
X = df[["size_diff_z", "loc_adv_z"]]
X = sm.add_constant(X)
y = df["win"]

model = sm.Logit(y, X)
result = model.fit(disp=False)

print("Logit results (standardized predictors):")
print(result.summary())

# Also show unstandardized coefficients for interpretability
X_un = df[["size_diff", "loc_adv"]]
X_un = sm.add_constant(X_un)
model_un = sm.Logit(y, X_un)
result_un = model_un.fit(disp=False)

print("\nLogit results (raw predictors):")
print(result_un.summary())

# Simple descriptive: win rate by size advantage and location advantage signs
size_adv = df["size_diff"] > 0
loc_adv = df["loc_adv"] > 0

summary = df.groupby([size_adv, loc_adv])["win"].mean().rename("win_rate").reset_index()
summary["size_adv"] = summary["size_diff"].map({False: "no", True: "yes"})
summary["loc_adv"] = summary["loc_adv"].map({False: "no", True: "yes"})
print("\nWin rates by advantages (size_adv, loc_adv):")
print(summary[["size_adv", "loc_adv", "win_rate"]].to_string(index=False))
